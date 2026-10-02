"""
src/train.py
============
학습·평가 파이프라인입니다. `python -m src.train` 한 줄로 전체 결과를 재현합니다.

데이터 분할
----------
- Train : Batch 1 중 hold-out 을 제외한 셀 → 정책 단위 5-fold CV 로 Train 성능 산출
- Valid : Batch 1 Hold-out — 충전 정책 단위로 분리해, 같은 정책 셀이 train/valid 에 섞이지 않도록 합니다.
- Test  : Batch 2 (필수, standard / newstructure 그룹별 함께 보고), Batch 3 (추가 검증)

절차
----
1. 하이퍼파라미터 : train 셀의 정책 단위 5-fold CV 로만 선택합니다 (valid·test 미사용).
2. Train 성능   : 선택된 파라미터의 CV fold 평균 MAPE
3. Valid 성능   : train 으로 fit → hold-out 9셀 MAPE
4. 모델 선택    : Batch 1 out-of-fold 41셀(train CV 32셀 + valid 9셀) MAPE 가 가장 낮은 모델
5. Test 성능    : Batch 1 전체(train + valid)로 다시 fit → Batch 2 / Batch 3 MAPE (테스트는 한 번만 평가)

결측 대체·스케일링·모델을 하나의 Pipeline 으로 묶어, 모든 전처리가 학습 데이터에서만 fit 되도록 합니다.

지표
----
Target 은 log10(cycle_life) 이며, 예측값을 10^ŷ 로 되돌린 사이클 단위 MAPE(%)로 평가합니다.
Gap 은 오차 증가량(뒤 단계 MAPE − 앞 단계 MAPE)이며, (+) 는 성능 저하입니다.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features import FEATURE_SETS, TARGET, make_dataset
from src.preprocess import PROCESSED_DIR, load_processed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

SPLIT_ORDER = ["train", "valid", "test_b2", "test_b3"]
PAPER_TARGET_MAPE = 9.1        # 원논문 회귀 모델 테스트 MAPE (%)
N_FOLDS = 5
SEED = 42


# ---------------------------------------------------------------------------
# 데이터 분할
# ---------------------------------------------------------------------------
def policy_holdout_split(df: pd.DataFrame, step: int = 4, offset: int = 2,
                         group_col: str = "policy_base", target: str = TARGET) -> pd.Series:
    """
    Batch 1 을 충전 정책 단위로 train / valid 로 나눕니다.

    정책을 평균 수명 순으로 정렬한 뒤 step 개마다 1개(offset 번째)를 valid 로 보내는 계통 추출입니다.
      - 같은 정책 셀은 반드시 같은 쪽에 들어갑니다 (정책 정보 누설 방지).
      - valid 정책이 수명 범위 전체에 고르게 퍼지고, 결과가 재현 가능합니다.
      - 최장·최단 정책은 train 에 남아 학습 수명 범위를 넓게 유지합니다 (테스트는 외삽 문제).
    """
    split = pd.Series(np.select([df["batch"] == 2, df["batch"] == 3], ["test_b2", "test_b3"], "train"),
                      index=df.index, name="split")
    b1 = df[df["batch"] == 1]
    order = b1.groupby(group_col)[target].mean().sort_values().index
    valid_policies = [p for i, p in enumerate(order) if i % step == offset]
    split[(df["batch"] == 1) & df[group_col].isin(valid_policies)] = "valid"
    return split


def policy_folds(df: pd.DataFrame, n_folds: int = 5, group_col: str = "policy_base",
                 target: str = TARGET) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    train 셀의 정책 단위 CV fold 를 만듭니다 (위치 인덱스 기준).

    정책을 평균 수명 순으로 정렬해 fold 0,1,…,n-1 에 번갈아 배정합니다.
      - 같은 정책 셀은 같은 fold 에 들어갑니다 (정책 정보 누설 방지).
      - 각 fold 에 짧은·긴 수명 정책이 고르게 들어갑니다.
      - sklearn 버전과 무관하게 fold 가 같아 결과가 재현됩니다 (GroupKFold 는 버전마다 배정이 달라질 수 있음).
    """
    order = df.groupby(group_col)[target].mean().sort_values().index
    fold_of = {p: i % n_folds for i, p in enumerate(order)}
    f = df[group_col].map(fold_of).to_numpy()
    return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in range(n_folds)]


def load_dataset(processed_dir: Path | str = PROCESSED_DIR) -> pd.DataFrame:
    """전처리 캐시 → 모델링 데이터셋 + split 컬럼."""
    cells, summary, qd = load_processed(processed_dir)
    data = make_dataset(cells, summary, qd, n_cycles=100)
    data["split"] = policy_holdout_split(data)
    return data


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
def mape_cycles(y_log_true, y_log_pred) -> float:
    """log10 수명 예측을 사이클로 되돌린 MAPE (%)."""
    yt, yp = 10 ** np.asarray(y_log_true), 10 ** np.asarray(y_log_pred)
    return float(np.mean(np.abs(yp - yt) / yt) * 100)


MAPE_SCORER = make_scorer(mape_cycles, greater_is_better=False)


# ---------------------------------------------------------------------------
# 후보 모델 (Day 1 모델 설계 전략 4.4)
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    name: str
    feature_set: str
    estimator: object
    grid: dict = field(default_factory=dict)
    role: str = ""

    @property
    def features(self) -> list[str]:
        return FEATURE_SETS[self.feature_set]

    def pipeline(self) -> Pipeline:
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),   # F3 계열의 IR_min 결측 (B2 6셀)
            ("scale", StandardScaler()),
            ("model", clone(self.estimator)),
        ])


EN_GRID = {"model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "model__l1_ratio": [0.1, 0.5, 0.9]}

CANDIDATES = [
    # 주 실험 : Day 1 후보 모델
    Candidate("LinearRegression", "F1", LinearRegression(), role="베이스라인 (원논문 Variance model)"),
    Candidate("ElasticNet", "F2", ElasticNet(max_iter=50_000, random_state=SEED), EN_GRID, role="1순위 : 외삽 가능 + 정규화"),
    Candidate("Huber", "F2", HuberRegressor(max_iter=5_000),
              {"model__epsilon": [1.35, 1.7, 2.0], "model__alpha": [1e-4, 1e-2, 1e-1, 1.0]}, role="긴 꼬리에 강건한 선형"),
    Candidate("RandomForest", "F2", RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
              {"model__max_depth": [None, 3, 5], "model__min_samples_leaf": [1, 2, 4]}, role="비교군 : 비선형, 외삽 불가"),
    Candidate("XGBoost", "F2", XGBRegressor(n_estimators=300, learning_rate=0.05, subsample=0.8,
                                              random_state=SEED, n_jobs=1, verbosity=0),
              {"model__max_depth": [2, 3], "model__min_child_weight": [1, 3]}, role="비교군 : 비선형, 외삽 불가"),
    # 피처셋 비교 실험 : 같은 모델(ElasticNet)로 피처셋만 바꿈
    Candidate("ElasticNet", "F1", ElasticNet(max_iter=50_000, random_state=SEED), EN_GRID, role="피처셋 비교 : F1"),
    Candidate("ElasticNet", "F3", ElasticNet(max_iter=50_000, random_state=SEED), EN_GRID, role="피처셋 비교 : 배치 과적합 확인"),
    Candidate("ElasticNet", "F3+Policy", ElasticNet(max_iter=50_000, random_state=SEED), EN_GRID, role="피처셋 비교 : 정책 피처 과적합 확인"),
]


# ---------------------------------------------------------------------------
# 학습 · 평가
# ---------------------------------------------------------------------------
def evaluate_candidate(cand: Candidate, data: pd.DataFrame) -> tuple[dict, pd.DataFrame, Pipeline]:
    tr = data[data.split == "train"]
    va = data[data.split == "valid"]
    b1 = data[data.batch == 1]
    X, y = tr[cand.features], tr[TARGET]
    cv = policy_folds(tr, N_FOLDS)

    # 1) 하이퍼파라미터 : train 정책 단위 CV 로만 선택
    if cand.grid:
        gs = GridSearchCV(cand.pipeline(), cand.grid, scoring=MAPE_SCORER, cv=cv, n_jobs=1)
        gs.fit(X, y)
        params = gs.best_params_
    else:
        params = {}

    # 2) Train (Batch 1 CV) : fold 별 MAPE 평균 + out-of-fold 예측
    fold_mape, oof = [], pd.Series(np.nan, index=tr.index)
    for a, b in cv:
        m = cand.pipeline().set_params(**params).fit(X.iloc[a], y.iloc[a])
        oof.iloc[b] = m.predict(X.iloc[b])
        fold_mape.append(mape_cycles(y.iloc[b], oof.iloc[b]))

    # 3) Valid (Batch 1 Hold-out)
    m_tr = cand.pipeline().set_params(**params).fit(X, y)
    valid_pred = pd.Series(m_tr.predict(va[cand.features]), index=va.index)
    valid_mape = mape_cycles(va[TARGET], valid_pred)

    # 4) Test : Batch 1 전체로 다시 fit → Batch 2 / Batch 3
    final = cand.pipeline().set_params(**params).fit(b1[cand.features], b1[TARGET])

    # 셀별 예측 : train = out-of-fold, valid = train 모델, test = Batch 1 전체 모델 (모두 학습에 쓰이지 않은 셀)
    pred = data[["cell_key", "batch", "group", "policy_base", "split", "cycle_life"]].copy()
    pred["y_pred_log"] = final.predict(data[cand.features])
    pred.loc[oof.index, "y_pred_log"] = oof
    pred.loc[valid_pred.index, "y_pred_log"] = valid_pred
    pred["pred_life"] = 10 ** pred["y_pred_log"]
    pred["ape"] = (pred.pred_life - pred.cycle_life).abs() / pred.cycle_life * 100
    pred.insert(0, "model", f"{cand.name} / {cand.feature_set}")

    def m(mask):
        return float(pred.loc[mask, "ape"].mean())

    row = {
        "model": f"{cand.name} / {cand.feature_set}", "algorithm": cand.name, "feature_set": cand.feature_set,
        "role": cand.role, "params": json.dumps({k.replace("model__", ""): v for k, v in params.items()}),
        "train_cv_mape": float(np.mean(fold_mape)), "train_cv_std": float(np.std(fold_mape)),
        "valid_mape": valid_mape,
        "b1_oof_mape": m(pred.batch == 1),
        "test_b2_mape": m(pred.split == "test_b2"),
        "test_b2_standard_mape": m(pred.group == "B2-standard"),
        "test_b2_newstructure_mape": m(pred.group == "B2-newstructure"),
        "test_b3_mape": m(pred.split == "test_b3"),
    }
    row["gap_train_valid"] = row["valid_mape"] - row["train_cv_mape"]
    row["gap_valid_test"] = row["test_b2_mape"] - row["valid_mape"]
    row["gap_target_test"] = row["test_b2_mape"] - PAPER_TARGET_MAPE
    row["gap_b2_b3"] = row["test_b3_mape"] - row["test_b2_mape"]
    row["gap_target_test_b3"] = row["test_b3_mape"] - PAPER_TARGET_MAPE
    return row, pred, final


def coefficients(model: Pipeline, features: list[str]) -> pd.Series | None:
    """선형 모델의 표준화 계수 (피처 1σ 변화당 log10 수명 변화)."""
    est = model.named_steps["model"]
    if hasattr(est, "coef_"):
        return pd.Series(np.ravel(est.coef_), index=features)
    return None


def select_best(perf: pd.DataFrame) -> str:
    """
    최종 모델 선택 : Batch 1 데이터만 사용합니다 (Batch 2·3 미사용).
    Batch 1 의 모든 셀(41)에 대한 out-of-fold MAPE 가 가장 낮은 모델을 고릅니다.
    valid 단독(9셀·5정책)은 표본이 작아 CV 와 크게 어긋날 수 있으므로(예: ElasticNet / F3+Policy Valid 7.7% vs CV 10.0%)
    train CV 32셀과 valid 9셀을 합친 41셀 기준을 사용합니다.
    """
    return perf.sort_values(["b1_oof_mape", "valid_mape"]).iloc[0]["model"]


def run(out_dir: Path | str = RESULTS_DIR, processed_dir: Path | str = PROCESSED_DIR, verbose: bool = True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(processed_dir)

    rows, preds, coefs = [], [], {}
    for cand in CANDIDATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row, pred, final = evaluate_candidate(cand, data)
        rows.append(row); preds.append(pred)
        c = coefficients(final, cand.features)
        if c is not None:
            coefs[row["model"]] = c
        if verbose:
            print(f"{row['model']:<26} CV {row['train_cv_mape']:5.1f}  Valid {row['valid_mape']:5.1f}  B1-OOF {row['b1_oof_mape']:5.1f}  "
                  f"B2 {row['test_b2_mape']:5.1f} (std {row['test_b2_standard_mape']:5.1f} / new {row['test_b2_newstructure_mape']:5.1f})  "
                  f"B3 {row['test_b3_mape']:5.1f}")

    perf = pd.DataFrame(rows)
    best = select_best(perf)
    perf["selected"] = perf["model"] == best
    perf.to_csv(out_dir / "model_performance.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(coefs).to_csv(out_dir / "coefficients.csv")
    (out_dir / "split.json").write_text(json.dumps(
        {s: data.loc[data.split == s, "cell_key"].tolist() for s in SPLIT_ORDER}, indent=1))
    if verbose:
        print(f"\n선택 모델 (Batch 1 OOF 기준) : {best}")
        print(f"저장 : {out_dir}")
    return perf, data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="ESS 배터리 수명 회귀 모델 학습·평가")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = ap.parse_args(argv)
    run(args.out_dir, args.processed_dir)


if __name__ == "__main__":
    main()
