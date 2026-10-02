"""
src/features.py
===============
EDA(01_EDA.ipynb)와 모델링에서 함께 사용하는 피처 계산 함수입니다.
입력은 `src.preprocess.load_processed()` 가 반환하는 (cells, summary, qd) 입니다.

설계
----
태스크 : Regression (target = log10(cycle_life), 원논문 목표 MAPE 9.1%)
관측 창은 초기 100 사이클이며, ΔQ_100-10(V) 통계를 핵심 피처로 사용합니다.
관측 창(n_cycles)을 인자로 받으므로 다른 창 크기와의 비교에도 같은 함수를 사용할 수 있습니다.

    from src.preprocess import load_processed
    from src.features import build_feature_table, delta_q_matrix, knee_table, vif

    cells, summary, qd = load_processed()
    feats = build_feature_table(cells, summary, qd, n_cycles=100)   # 기본 ΔQ 쌍 = (100, 10)
    knees = knee_table(summary)

사이클 인덱스
------------
preprocess.py 와 같은 cycle_idx(0부터 시작, 원논문 코드 기준)를 사용합니다.
Batch 1 의 cycle_idx 0 은 데이터가 없으므로, 요약 피처는 cycle_idx >= 1 만 사용합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

EOL_AH = 0.88
EPS = 1e-12
DENOISE_WINDOW = 9          # ΔQ 단일 점 스파이크 제거용 중앙값 필터 폭 (전압축 1,000점 중 9점)


# ---------------------------------------------------------------------------
# ΔQ(V)
# ---------------------------------------------------------------------------
def align_start(Q: np.ndarray) -> np.ndarray:
    """
    Qdlin 의 시작점(전압축 첫 점, 고전압 쪽) 값을 0으로 맞춥니다.
    배치별 커브 시작점 차이로 Qdlin 을 직접 비교할 때 생기는 왜곡을 보정하는 옵션입니다.
    """
    return Q - Q[..., :1]


def delta_q_matrix(qd: dict, i: int, j: int, align: bool = False, denoise: bool = False) -> np.ndarray:
    """
    모든 셀의 ΔQ_i-j(V) = Qdlin[i] - Qdlin[j].  shape = (n_cells, 1000)
    denoise=True 이면 전압축 중앙값 필터로 단일 점 측정 스파이크를 제거합니다 (예: b1c41 의 3.1V 스파이크).
    """
    Q = qd["Qdlin"]
    if max(i, j) > int(qd["cycles"].max()):
        raise ValueError(f"cycle_idx {max(i, j)} 는 저장 범위 밖입니다 (0~{qd['cycles'].max()}).")
    qi, qj = Q[:, i].astype(float), Q[:, j].astype(float)
    if align:
        qi, qj = align_start(qi), align_start(qj)
    dq = qi - qj
    if denoise:
        dq = median_filter(dq, size=(1, DENOISE_WINDOW), mode="nearest")
    return dq


def delta_q(qd: dict, cell_key: str, i: int, j: int, align: bool = False) -> np.ndarray:
    """단일 셀의 ΔQ_i-j(V).  shape = (1000,)"""
    return delta_q_matrix(qd, i, j, align)[qd["index"][cell_key]]


def dq_stats(dq: np.ndarray, prefix: str = "dq") -> dict:
    """
    ΔQ(V) 곡선 1개 → 통계 피처.
    원논문은 ΔQ_100-10 의 log10(var) 하나로 수명과 강한 선형 관계를 보였습니다.
    min / mean / skew / kurt 는 원논문 Discharge·Full 모델에서 사용한 보조 통계입니다.
    """
    x = dq[np.isfinite(dq)]
    if x.size < 10:
        keys = ["var", "logvar", "min", "logabsmin", "mean", "skew", "kurt"]
        return {f"{prefix}_{k}": np.nan for k in keys}
    mu, sd = x.mean(), x.std()
    skew = ((x - mu) ** 3).mean() / (sd ** 3 + EPS)
    kurt = ((x - mu) ** 4).mean() / (sd ** 4 + EPS)
    var = x.var()
    return {
        f"{prefix}_var": var,
        f"{prefix}_logvar": np.log10(var + EPS),
        f"{prefix}_min": x.min(),
        f"{prefix}_logabsmin": np.log10(abs(x.min()) + EPS),
        f"{prefix}_mean": mu,
        f"{prefix}_skew": skew,
        f"{prefix}_kurt": kurt,
    }


def dq_feature_frame(qd: dict, pairs: list[tuple[int, int]], align: bool = False,
                     denoise: bool = False) -> pd.DataFrame:
    """여러 (i, j) 쌍에 대해 셀별 ΔQ 통계 피처 표."""
    out = pd.DataFrame({"cell_key": qd["cell_keys"]})
    for i, j in pairs:
        M = delta_q_matrix(qd, i, j, align, denoise)
        stats = pd.DataFrame([dq_stats(row, prefix=f"dq{i}_{j}") for row in M])
        out = pd.concat([out, stats], axis=1)
    return out


# ---------------------------------------------------------------------------
# summary 기반 초기 사이클 피처
# ---------------------------------------------------------------------------
def _window(summary: pd.DataFrame, start: int, end: int, drop_outlier: bool = True) -> pd.DataFrame:
    s = summary[(summary["cycle_idx"] >= start) & (summary["cycle_idx"] <= end)]
    if drop_outlier and "qd_outlier" in s:
        s = s[~s["qd_outlier"]]
    return s


def summary_features(summary: pd.DataFrame, n_cycles: int, start: int = 1) -> pd.DataFrame:
    """
    초기 n_cycles 사이클(cycle_idx start ~ n_cycles) 요약 피처.
    원논문 피처 정의를 관측 창 크기에 맞게 일반화했습니다.

    QD     : 첫 사이클 용량, 최대-첫 용량 차이, 선형 기울기/절편 (용량 감소 추세)
    IR     : 최소값, 마지막-첫 사이클 차이
    T      : Tmax 최대, Tavg 평균, Tavg 적분(사이클 합)
    시간   : 충전시간 평균·중앙값, 방전시간 평균
             (중앙값은 시험 일시정지로 1개 사이클이 수천 분으로 기록되는 시간 공백에 강건합니다)
    """
    s = _window(summary, start, n_cycles)
    rows = []
    for key, g in s.groupby("cell_key", sort=False):
        g = g.sort_values("cycle_idx")
        x, qd = g["cycle_idx"].to_numpy(float), g["QD"].to_numpy(float)
        ok = np.isfinite(qd)
        if ok.sum() >= 2:
            slope, intercept = np.polyfit(x[ok], qd[ok], 1)
        else:
            slope = intercept = np.nan
        ir = g["IR"].replace(0, np.nan)            # IR=0 은 미측정으로 보고 결측 처리 (Batch 2 의 약 18%)
        rows.append({
            "cell_key": key,
            "QD_first": qd[ok][0] if ok.any() else np.nan,
            "QD_max_minus_first": (np.nanmax(qd) - qd[ok][0]) if ok.any() else np.nan,
            "QD_slope": slope,
            "QD_intercept": intercept,
            "IR_min": ir.min(),
            "IR_delta": ir.dropna().iloc[-1] - ir.dropna().iloc[0] if ir.notna().sum() >= 2 else np.nan,
            "Tmax_max": g["Tmax"].max(),
            "Tavg_mean": g["Tavg"].mean(),
            "Tavg_integral": g["Tavg"].sum(),
            "chargetime_mean": g["chargetime"].mean(),
            "chargetime_median": g["chargetime"].median(),
            "discharge_time_mean": g["discharge_time"].mean() if "discharge_time" in g else np.nan,
            "n_valid_cycles": int(ok.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 최종 피처 테이블
# ---------------------------------------------------------------------------
POLICY_COLS = ["C1", "Q1", "C2", "C_avg_0_80"]
META_COLS = ["cell_key", "batch", "policy", "policy_group", "cycle_life", "label"]


def build_feature_table(cells: pd.DataFrame, summary: pd.DataFrame, qd: dict,
                        n_cycles: int, dq_pairs: list[tuple[int, int]] | None = None,
                        align: bool = False, include_policy: bool = True, dq_denoise: bool = False) -> pd.DataFrame:
    """
    셀 1개 = 1행 피처 테이블.

    n_cycles : 관측 창 (기본 사용 = 100, 원논문 회귀 모델 기준)
    dq_pairs : ΔQ 쌍 목록. None 이면 창에 맞는 기본값 (n=100 → (100,10), n<=10 → (n, n-1))
               ※ 모든 쌍은 cycle_idx <= n_cycles 여야 합니다 (미래 정보 누설 방지)
    """
    if dq_pairs is None:
        dq_pairs = [(100, 10)] if n_cycles >= 100 else [(n_cycles, max(n_cycles - 1, 1))]
    bad = [p for p in dq_pairs if max(p) > n_cycles]
    if bad:
        raise ValueError(f"ΔQ 쌍 {bad} 가 관측 창(n_cycles={n_cycles})을 넘습니다.")

    meta = cells[[c for c in META_COLS + (POLICY_COLS if include_policy else []) if c in cells]]
    feats = (meta
             .merge(summary_features(summary, n_cycles), on="cell_key", how="left")
             .merge(dq_feature_frame(qd, dq_pairs, align, dq_denoise), on="cell_key", how="left"))
    feats.attrs["n_cycles"] = n_cycles
    feats.attrs["dq_pairs"] = dq_pairs
    return feats


def feature_columns(df: pd.DataFrame) -> list[str]:
    """모델 입력으로 쓸 수 있는 숫자형 피처 컬럼 (메타/타깃 제외)."""
    drop = set(META_COLS) | {"n_valid_cycles"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# 열화 곡선: knee point / 열화 가속
# ---------------------------------------------------------------------------
def _smooth(y: np.ndarray, w: int = 15) -> np.ndarray:
    return pd.Series(y).rolling(w, center=True, min_periods=1).median().to_numpy()


def knee_point(cycle: np.ndarray, qd: np.ndarray, min_seg: int = 20) -> dict:
    """
    2구간 선형 회귀(piecewise linear)로 knee point 를 탐색합니다.
    모든 분할점 k 에 대해 [0:k], [k:n] 을 각각 직선으로 맞추고 SSE 합이 최소인 k 를 knee 로 정합니다.

    returns: knee_cycle, slope_before, slope_after (Ah/cycle), accel_ratio = after / before
    """
    m = np.isfinite(qd)
    x, y = np.asarray(cycle, float)[m], _smooth(np.asarray(qd, float)[m])
    n = len(x)
    nan = {"knee_cycle": np.nan, "slope_before": np.nan, "slope_after": np.nan, "accel_ratio": np.nan}
    if n < 2 * min_seg:
        return nan

    # 누적합으로 모든 분할점의 SSE 를 O(n) 에 계산
    cx, cy = np.cumsum(x), np.cumsum(y)
    cxx, cxy, cyy = np.cumsum(x * x), np.cumsum(x * y), np.cumsum(y * y)

    def sse(a, b):  # 구간 [a, b) 의 선형 회귀 SSE, slope
        k = b - a
        sx = cx[b - 1] - (cx[a - 1] if a else 0)
        sy = cy[b - 1] - (cy[a - 1] if a else 0)
        sxx = cxx[b - 1] - (cxx[a - 1] if a else 0)
        sxy = cxy[b - 1] - (cxy[a - 1] if a else 0)
        syy = cyy[b - 1] - (cyy[a - 1] if a else 0)
        vx = sxx - sx * sx / k
        cov = sxy - sx * sy / k
        vy = syy - sy * sy / k
        slope = cov / vx if vx > EPS else 0.0
        return vy - slope * cov, slope

    best = (np.inf, None, None, None)
    for k in range(min_seg, n - min_seg + 1):
        e1, s1 = sse(0, k)
        e2, s2 = sse(k, n)
        if e1 + e2 < best[0]:
            best = (e1 + e2, k, s1, s2)
    _, k, s1, s2 = best
    return {
        "knee_cycle": float(x[k]),
        "slope_before": s1,
        "slope_after": s2,
        "accel_ratio": s2 / s1 if abs(s1) > EPS else np.nan,
    }


def knee_table(summary: pd.DataFrame, until_eol: bool = True, x_col: str = "cycle") -> pd.DataFrame:
    """
    셀별 knee point / 열화 가속 지표.
    until_eol=True 이면 QD 가 처음 EOL(0.88Ah) 아래로 내려가는 지점까지만 사용합니다.
    """
    rows = []
    s = summary[~summary["qd_outlier"]] if "qd_outlier" in summary else summary
    s = s[s["cycle_idx"] >= 1]
    for key, g in s.groupby("cell_key", sort=False):
        g = g.sort_values("cycle_idx")
        if until_eol:
            below = np.flatnonzero(g["QD"].to_numpy() < EOL_AH)
            if below.size:
                g = g.iloc[: below[0] + 1]
        rows.append({"cell_key": key, **knee_point(g[x_col].to_numpy(), g["QD"].to_numpy())})
    return pd.DataFrame(rows)


def fade_rate(summary: pd.DataFrame, windows: list[tuple[int, int]] = ((1, 100), (100, 200), (200, 300))
              ) -> pd.DataFrame:
    """구간별 용량 감소 기울기 (Ah/cycle). 열화 속도의 일정/가속 여부를 비교합니다."""
    out = []
    s = summary[~summary["qd_outlier"]] if "qd_outlier" in summary else summary
    for key, g in s.groupby("cell_key", sort=False):
        row = {"cell_key": key}
        for a, b in windows:
            w = g[(g["cycle_idx"] >= a) & (g["cycle_idx"] <= b)]
            ok = w["QD"].notna()
            row[f"fade_{a}_{b}"] = (np.polyfit(w.loc[ok, "cycle_idx"], w.loc[ok, "QD"], 1)[0]
                                    if ok.sum() >= 5 else np.nan)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 다중공선성
# ---------------------------------------------------------------------------
def vif(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    VIF_i = 1 / (1 - R²_i).  10 이상이면 다중공선성 의심.
    numpy 최소제곱으로 계산합니다.
    """
    cols = cols or feature_columns(df)
    X = df[cols].astype(float).dropna()
    X = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    X = X.dropna(axis=1)
    res = []
    for c in X.columns:
        y = X[c].to_numpy()
        A = np.column_stack([np.ones(len(X)), X.drop(columns=c).to_numpy()])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / (((y - y.mean()) ** 2).sum() + EPS)
        res.append({"feature": c, "VIF": 1 / max(1 - r2, EPS)})
    return pd.DataFrame(res).sort_values("VIF", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Day 1 모델 설계 전략의 피처셋 / 모델링 데이터셋
# ---------------------------------------------------------------------------
TARGET = "log_life"          # log10(cycle_life)

FEATURE_SETS = {
    # F1 : 세 배치 공통 최강 신호 (원논문 Variance model)
    "F1": ["dq100_10_logvar"],
    # F2 : 배치 간 상관 부호가 일관된 ΔQ 통계 (주 피처셋)
    "F2": ["dq100_10_logvar", "dq100_10_skew", "dq100_10_kurt"],
    # F3 : F2 + Batch 1 에서만 유의하고 배치 간 부호가 바뀌는 측정 피처 (배치 과적합 확인용 비교군)
    "F3": ["dq100_10_logvar", "dq100_10_skew", "dq100_10_kurt", "chargetime_median",
           "QD_slope", "QD_max_minus_first", "IR_min", "Tavg_mean"],
    # F3+Policy : F3 + 충전 정책 설정값 (정책 피처 과적합 확인용 비교군)
    "F3+Policy": ["dq100_10_logvar", "dq100_10_skew", "dq100_10_kurt", "chargetime_median",
                  "QD_slope", "QD_max_minus_first", "IR_min", "Tavg_mean",
                  "C_avg_0_80", "C1", "Q1", "C2"],
}


def make_dataset(cells: pd.DataFrame, summary: pd.DataFrame, qd: dict, n_cycles: int = 100) -> pd.DataFrame:
    """
    셀 단위 모델링 데이터셋 : 피처(초기 n_cycles 사이클) + 타깃 + 분석 그룹.
      - group       : B1 / B2-standard / B2-newstructure / B3
      - policy_base : 시험 구조 접미사(-newstructure)를 뗀 충전 정책 (정책 단위 분할의 그룹 키)
      - log_life    : log10(cycle_life)
    ΔQ 통계는 단일 점 스파이크를 제거한(denoise) 곡선으로 계산합니다.
    """
    feat = build_feature_table(cells, summary, qd, n_cycles=n_cycles, dq_denoise=True)
    extra = cells[["cell_key", "curve_truncated"]] if "curve_truncated" in cells else cells[["cell_key"]]
    feat = feat.merge(extra, on="cell_key", how="left")
    feat["group"] = np.where(feat["batch"] == 2, "B2-" + feat["policy_group"], "B" + feat["batch"].astype(str))
    feat["policy_base"] = feat["policy"].str.replace("-newstructure", "", regex=False)
    feat[TARGET] = np.log10(feat["cycle_life"])
    return feat
