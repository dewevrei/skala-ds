"""
src/preprocess.py
=================
MIT-Stanford 배터리 데이터(.mat, MATLAB v7.3 / HDF5)를 분석용 캐시 파일로 변환합니다.

원본 3개(약 7.7GB)를 매번 메모리에 올리지 않도록 h5py로 필요한 필드만 읽어
`data/processed/`에 저장합니다. 노트북과 모델 코드는 이 캐시만 읽습니다.

실행
----
    python -m src.preprocess                      # data/raw → data/processed
    python -m src.preprocess --max-cycle 100      # Qdlin 저장 범위 (cycle_idx 0~N)
    python -m src.preprocess --no-discharge-time  # 방전 시간 계산 생략

산출물 (data/processed/)
-----------------------
    cells.parquet    셀 단위 : 메타데이터, 충전 정책 파싱, 수명 보정, 제외 여부와 사유
    summary.parquet  사이클 단위 : QD, QC, IR, Tavg/Tmin/Tmax, chargetime, discharge_time
    qdlin.npz        cycle_idx 0~max_cycle 의 Qdlin / Tdlin (셀 × 사이클 × 1000) + 전압축 V
    meta.json        실행 설정, 입력 파일, 생성 시각

사용 예
------
    from src.preprocess import load_processed, get_curve
    cells, summary, qd = load_processed()                                   # 분석용 (제외 셀 제외)
    cells_all, summary_all, qd_all = load_processed(include_excluded=True)  # 제외 셀 포함

데이터 정제 규칙
---------------
근거 : 원논문 공식 코드 (github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

1. Batch 1 (2017-05-12) 제외 : b1c8, b1c10, b1c12, b1c13, b1c22 — 80% 용량(EOL) 도달 전 시험 중단
2. Batch 1 수명 보정 : b1c0~b1c4 는 원논문 Batch 2(2017-06-30)로 이어서 측정된 셀입니다.
   원논문과 같이 cycle_life 에 이어진 사이클 수(add_len)를 더해 보정합니다.
   초기 사이클 데이터는 Batch 1 파일에 온전히 있으므로 수명 레이블만 보정하고,
   열화 곡선은 Batch 1 구간까지만 있으므로 curve_truncated=True 로 표시합니다.
3. Batch 3 (2018-04-12) 제외 : b3c2, b3c23, b3c32, b3c37, b3c42, b3c43 — 노이즈 채널
4. cycle_life 가 없는 셀 제외 (2018-02-20 파일의 VarCharge / SLOWCYCLE 셀)
5. 원논문 Batch 2 파일(2017-06-30)을 사용할 때만 b2c7/8/9/15/16 을 중복으로 제외합니다.
   과제의 Batch 2(2018-02-20)는 원논문 Batch 2 와 다른 배치이므로 이 규칙을 적용하지 않습니다.

제외한 셀도 삭제하지 않고 cells.parquet 에 exclude / exclude_reason 으로 기록합니다.

사이클 인덱스
------------
- cycle_idx : 파일 내 사이클 위치(0부터 시작). 원논문 코드의 cycles['10'], cycles['100'] 과 같은 기준입니다.
              ΔQ_100-10(V) = Qdlin[cycle_idx=100] - Qdlin[cycle_idx=10]
- cycle     : summary['cycle'] 에 기록된 실제 사이클 번호
- Batch 1 의 cycle_idx 0 (1번 사이클)은 원본에 데이터가 없어 QD = 0 으로 기록되어 있습니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm 미설치 시 진행률 표시 생략
    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# 경로 / 상수
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NOMINAL_AH = 1.1                 # A123 APR18650M1A 공칭 용량
EOL_AH = 0.8 * NOMINAL_AH        # EOL = 공칭 용량의 80% = 0.88 Ah
LABEL_THRESHOLD = 550            # 장/단수명 분류 기준 (참고용 label 컬럼)
N_VPOINTS = 1000                 # Qdlin / Tdlin 길이
DEFAULT_MAX_CYCLE = 100          # Qdlin 을 저장할 최대 cycle_idx
DISCHARGE_I_THRESHOLD = -0.1     # 방전 구간 판정 전류 (A), I < -0.1

# 파일 날짜 → 배치 번호
FILE_BATCH = {
    "2017-05-12": 1,
    "2017-06-30": 2,   # 원논문 Batch 2 (과제 데이터에는 없음)
    "2018-02-20": 2,   # Batch 2 (테스트)
    "2018-04-12": 3,
}
SKIP_FILES = ("2018-04-03",)     # varcharge : 다른 논문(Attia 2020)용, 사용하지 않음

# 원논문 공식 코드 기준 정제 규칙
B1_NOT_REACH_EOL = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
B3_NOISY = ["b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"]
B1_CARRYOVER = {            # batch1 셀 : (원논문 batch2 셀, 이어진 사이클 수)
    "b1c0": ("b2c7", 662),
    "b1c1": ("b2c8", 981),
    "b1c2": ("b2c9", 1060),
    "b1c3": ("b2c15", 208),
    "b1c4": ("b2c16", 482),
}
ORIGINAL_B2_FILE = "2017-06-30"

# summary 필드명 → 컬럼명
SUMMARY_FIELDS = {
    "cycle": "cycle",
    "QDischarge": "QD",
    "QCharge": "QC",
    "IR": "IR",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "Tmax": "Tmax",
    "chargetime": "chargetime",
}

# summary QD 이상치 판정
QD_OUTLIER_ABS = (0.0, 1.3)      # 이 범위를 벗어나면 이상치 (Ah)
QD_OUTLIER_DEV = 0.05            # rolling median 대비 편차 (Ah)
QD_ROLLING_WINDOW = 9


# ---------------------------------------------------------------------------
# HDF5 (MATLAB v7.3) 읽기 헬퍼
# ---------------------------------------------------------------------------
def _is_matlab_empty(ds: h5py.Dataset) -> bool:
    return bool(ds.attrs.get("MATLAB_empty", 0))


def _read_array(f: h5py.File, ref) -> np.ndarray:
    """object reference → 1차원 float 배열 (MATLAB 빈 배열은 길이 0)."""
    ds = f[ref]
    if _is_matlab_empty(ds):
        return np.array([], dtype=float)
    return np.ravel(ds[()]).astype(float)


def _read_scalar(f: h5py.File, ref) -> float:
    arr = _read_array(f, ref)
    return float(arr[0]) if arr.size else np.nan


def _read_str(f: h5py.File, ref) -> str:
    """MATLAB char(uint16) → str."""
    ds = f[ref]
    if _is_matlab_empty(ds):
        return ""
    arr = np.ravel(ds[()])
    if arr.dtype.kind in "uif":
        return "".join(chr(int(c)) for c in arr if int(c) != 0).strip()
    if arr.dtype.kind in "SO":
        return b"".join(arr).decode(errors="ignore").strip()
    return str(arr).strip()


def _read_field_str(f, batch_grp, name, i) -> str | None:
    if name not in batch_grp:
        return None
    try:
        return _read_str(f, batch_grp[name][i, 0])
    except Exception:
        try:
            return str(_read_scalar(f, batch_grp[name][i, 0]))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 충전 정책 파싱
# ---------------------------------------------------------------------------
_POLICY_RE = re.compile(
    r"(?P<c1>\d+(?:\.\d+)?)C\((?P<q1>\d+(?:\.\d+)?)%\)-(?P<c2>\d+(?:\.\d+)?)C"
)


def parse_policy(policy: str) -> dict:
    """
    "4.8C(80%)-3.6C" → C1=4.8 (SOC 0→Q1 구간 C-rate), Q1=80 (전환 SOC %), C2=3.6 (Q1→80% 구간 C-rate)
    형식이 맞지 않으면 NaN 입니다. policy_group 으로 특수 실험(newstructure, varcharge, slowcycle)을 구분합니다.
    """
    p = policy or ""
    low = p.lower()
    if "varcharge" in low:
        group = "varcharge"
    elif "slowcycle" in low:
        group = "slowcycle"
    elif "newstructure" in low:
        group = "newstructure"
    elif _POLICY_RE.search(p):
        group = "standard"
    else:
        group = "other"

    m = _POLICY_RE.search(p)
    if m:
        c1, q1, c2 = float(m["c1"]), float(m["q1"]), float(m["c2"])
        # 0→80% SOC 평균 충전 C-rate (구간 가중 평균, 시간 기준)
        # t = Q1/C1 + (80-Q1)/C2  →  avg C = 80 / t
        if q1 < 80 and c2 > 0:
            t = q1 / c1 + (80 - q1) / c2
            c_avg = 80 / t
        else:
            c_avg = c1
    else:
        c1 = q1 = c2 = c_avg = np.nan

    return {"C1": c1, "Q1": q1, "C2": c2, "C_avg_0_80": c_avg, "policy_group": group}


# ---------------------------------------------------------------------------
# 배치 파일 1개 읽기
# ---------------------------------------------------------------------------
@dataclass
class BatchResult:
    batch: int
    file: str
    cells: list[dict] = field(default_factory=list)
    summaries: list[pd.DataFrame] = field(default_factory=list)
    qdlin: dict[str, np.ndarray] = field(default_factory=dict)   # cell_key → (n_cyc, 1000)
    tdlin: dict[str, np.ndarray] = field(default_factory=dict)
    vdlin: np.ndarray | None = None


def _discharge_time(t: np.ndarray, I: np.ndarray) -> float:
    """사이클 내 방전 구간(I < 임계값) 지속 시간 (t 와 같은 단위, 분)."""
    if t.size == 0 or t.size != I.size:
        return np.nan
    mask = I < DISCHARGE_I_THRESHOLD
    if not mask.any():
        return np.nan
    return float(t[mask].max() - t[mask].min())


def read_batch(path: Path, batch: int, max_cycle: int, with_discharge_time: bool) -> BatchResult:
    res = BatchResult(batch=batch, file=path.name)
    n_store = max_cycle + 1

    with h5py.File(path, "r") as f:
        b = f["batch"]
        n_cells = b["summary"].shape[0]

        for i in tqdm(range(n_cells), desc=f"Batch {batch} ({path.name[:10]})"):
            key = f"b{batch}c{i}"

            # --- 셀 메타 ---
            cycle_life_raw = _read_scalar(f, b["cycle_life"][i, 0])
            policy = _read_field_str(f, b, "policy_readable", i) or _read_field_str(f, b, "policy", i) or ""
            barcode = _read_field_str(f, b, "barcode", i)
            channel = _read_field_str(f, b, "channel_id", i)

            # --- summary ---
            sg = f[b["summary"][i, 0]]
            cols = {}
            for fld, col in SUMMARY_FIELDS.items():
                cols[col] = np.ravel(sg[fld][()]).astype(float) if fld in sg else None
            n_sum = max((len(v) for v in cols.values() if v is not None), default=0)
            for col, v in cols.items():
                if v is None:
                    cols[col] = np.full(n_sum, np.nan)
                elif len(v) != n_sum:   # 길이 불일치 시 NaN 패딩
                    pad = np.full(n_sum, np.nan)
                    pad[: len(v)] = v
                    cols[col] = pad

            # --- cycles ---
            cg = f[b["cycles"][i, 0]]
            n_cyc = cg["I"].shape[0]
            q_arr = np.full((n_store, N_VPOINTS), np.nan, dtype=np.float32)
            t_arr = np.full((n_store, N_VPOINTS), np.nan, dtype=np.float32)
            q_refs = cg["Qdlin"][()]
            t_refs = cg["Tdlin"][()] if "Tdlin" in cg else None
            for j in range(min(n_cyc, n_store)):
                q = _read_array(f, q_refs[j, 0])
                if q.size == N_VPOINTS:
                    q_arr[j] = q
                if t_refs is not None:
                    tt = _read_array(f, t_refs[j, 0])
                    if tt.size == N_VPOINTS:
                        t_arr[j] = tt

            # 방전 시간 (summary 에 없으므로 사이클 시계열 t, I 로 계산)
            dis_time = np.full(n_sum, np.nan)
            if with_discharge_time and "t" in cg and "I" in cg:
                tr, ir = cg["t"][()], cg["I"][()]
                for j in range(min(n_cyc, n_sum)):
                    dis_time[j] = _discharge_time(_read_array(f, tr[j, 0]), _read_array(f, ir[j, 0]))

            if res.vdlin is None:
                if "Vdlin" in b:
                    try:
                        v = _read_array(f, b["Vdlin"][i, 0])
                        res.vdlin = v if v.size == N_VPOINTS else None
                    except Exception:
                        res.vdlin = None

            sdf = pd.DataFrame(cols)
            sdf.insert(0, "cycle_idx", np.arange(n_sum))
            sdf.insert(0, "batch", batch)
            sdf.insert(0, "cell_key", key)
            sdf["discharge_time"] = dis_time
            res.summaries.append(sdf)
            res.qdlin[key] = q_arr
            res.tdlin[key] = t_arr

            res.cells.append({
                "cell_key": key,
                "batch": batch,
                "batch_file": path.name,
                "cell_idx": i,
                "barcode": barcode,
                "channel_id": channel,
                "policy": policy,
                **parse_policy(policy),
                "cycle_life_raw": cycle_life_raw,
                "n_cycles_summary": n_sum,
                "n_cycles_cycles": n_cyc,
            })

    return res


# ---------------------------------------------------------------------------
# 정제 / 플래그
# ---------------------------------------------------------------------------
def flag_qd_outliers(summary: pd.DataFrame) -> pd.Series:
    """QD 스파이크/비정상값 플래그 (값은 지우지 않고 표시만 합니다)."""
    def _flag(g: pd.DataFrame) -> pd.Series:
        qd = g["QD"]
        med = qd.rolling(QD_ROLLING_WINDOW, center=True, min_periods=1).median()
        lo, hi = QD_OUTLIER_ABS
        return (qd <= lo) | (qd > hi) | ((qd - med).abs() > QD_OUTLIER_DEV) | qd.isna()

    return summary.groupby("cell_key", group_keys=False, sort=False).apply(_flag)


def observed_eol(summary: pd.DataFrame) -> pd.Series:
    """QD 가 처음으로 EOL(0.88Ah) 미만이 되는 cycle_idx (이상치·cycle_idx 0 제외). 미도달이면 NaN."""
    s = summary[(~summary["qd_outlier"]) & (summary["cycle_idx"] > 0)]
    hit = s[s["QD"] < EOL_AH].groupby("cell_key")["cycle_idx"].min()
    return hit


def apply_rules(cells: pd.DataFrame, batch2_file: str | None) -> pd.DataFrame:
    c = cells.copy()
    c["cycle_life"] = c["cycle_life_raw"]
    c["life_corrected"] = False
    c["curve_truncated"] = False
    c["exclude"] = False
    c["exclude_reason"] = ""

    def _exclude(mask, reason):
        new = mask & ~c["exclude"]
        c.loc[new, "exclude"] = True
        c.loc[new, "exclude_reason"] = reason

    # 규칙 1 : Batch 1 EOL 미도달
    _exclude(c["cell_key"].isin(B1_NOT_REACH_EOL), "원논문 제외: 80% 용량(EOL) 미도달")

    # 규칙 2 : Batch 1 → 원논문 Batch 2 이어측정 셀 수명 보정
    for b1, (b2, add_len) in B1_CARRYOVER.items():
        m = c["cell_key"] == b1
        if m.any():
            c.loc[m, "cycle_life"] = c.loc[m, "cycle_life_raw"] + add_len
            c.loc[m, "life_corrected"] = True
            c.loc[m, "curve_truncated"] = True

    # 규칙 5 : 원논문 Batch 2 파일(2017-06-30)을 사용할 때만 중복 셀 제외
    if batch2_file and batch2_file.startswith(ORIGINAL_B2_FILE):
        dup = [v[0] for v in B1_CARRYOVER.values()]
        _exclude(c["cell_key"].isin(dup), "원논문 제외: Batch 1 셀의 이어측정분(중복)")

    # 규칙 3 : Batch 3 노이즈 채널
    _exclude(c["cell_key"].isin(B3_NOISY), "원논문 제외: 노이즈 채널")

    # 규칙 4 : 수명 레이블 없음 (먼저 적용된 제외 사유가 있으면 유지)
    _exclude(c["cycle_life"].isna(), "cycle_life 없음 (VarCharge/SLOWCYCLE 등)")

    c["label"] = pd.array(
        np.where(c["cycle_life"].notna(), (c["cycle_life"] >= LABEL_THRESHOLD).astype(int), -1),
        dtype="Int64",
    )
    c.loc[c["cycle_life"].isna(), "label"] = pd.NA
    return c


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------
def find_batch_files(raw_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for p in sorted(raw_dir.glob("*.mat")):
        date = p.name[:10]
        if date in SKIP_FILES:
            print(f"  skip  : {p.name} (varcharge, 사용하지 않음)")
            continue
        if date not in FILE_BATCH:
            print(f"  skip  : {p.name} (알 수 없는 파일)")
            continue
        bn = FILE_BATCH[date]
        if bn in found:
            raise ValueError(f"Batch {bn} 파일이 2개 있습니다: {found[bn].name}, {p.name}")
        found[bn] = p
    if not found:
        raise FileNotFoundError(f"{raw_dir} 에 .mat 파일이 없습니다.")
    return dict(sorted(found.items()))


def run(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR,
        max_cycle: int = DEFAULT_MAX_CYCLE, with_discharge_time: bool = True) -> None:
    t0 = time.time()
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"입력 : {raw_dir}")
    files = find_batch_files(raw_dir)
    for bn, p in files.items():
        print(f"  Batch {bn} : {p.name} ({p.stat().st_size / 1024**3:.1f} GB)")

    results = [read_batch(p, bn, max_cycle, with_discharge_time) for bn, p in files.items()]

    # --- cells ---
    cells = pd.DataFrame([c for r in results for c in r.cells])
    batch2_file = files[2].name if 2 in files else None
    cells = apply_rules(cells, batch2_file)

    # --- summary ---
    summary = pd.concat([s for r in results for s in r.summaries], ignore_index=True)
    summary["qd_outlier"] = flag_qd_outliers(summary).astype(bool)

    eol = observed_eol(summary)
    cells["cycle_life_observed"] = cells["cell_key"].map(eol)

    # --- Qdlin / Tdlin ---
    keys = cells["cell_key"].tolist()
    qd_all = {k: v for r in results for k, v in r.qdlin.items()}
    td_all = {k: v for r in results for k, v in r.tdlin.items()}
    Q = np.stack([qd_all[k] for k in keys])
    T = np.stack([td_all[k] for k in keys])
    vdlin = next((r.vdlin for r in results if r.vdlin is not None), None)
    v_source = "file(Vdlin)"
    if vdlin is None:
        vdlin = np.linspace(3.6, 2.0, N_VPOINTS)   # 원논문 코드 기준 전압축
        v_source = "linspace(3.6, 2.0, 1000)"

    # --- 저장 ---
    cells.to_parquet(out_dir / "cells.parquet", index=False)
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    np.savez_compressed(
        out_dir / "qdlin.npz",
        cell_keys=np.array(keys), cycles=np.arange(max_cycle + 1),
        Qdlin=Q, Tdlin=T, V=vdlin.astype(np.float32),
    )
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": {str(k): v.name for k, v in files.items()},
        "max_cycle": max_cycle,
        "discharge_time": with_discharge_time,
        "V_source": v_source,
        "EOL_Ah": EOL_AH,
        "label_threshold": LABEL_THRESHOLD,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print_report(cells, summary)
    print(f"\n저장 : {out_dir}  ({time.time() - t0:.0f}s)")


def print_report(cells: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("정제 리포트")
    print("=" * 70)
    for bn, g in cells.groupby("batch"):
        use = g[~g["exclude"]]
        cl = use["cycle_life"]
        print(f"\n[Batch {bn}] {g['batch_file'].iloc[0]}")
        print(f"  전체 셀 {len(g)} → 사용 {len(use)} (제외 {g['exclude'].sum()})")
        for reason, gg in g[g["exclude"]].groupby("exclude_reason"):
            print(f"    - {reason}: {', '.join(gg['cell_key'])}")
        if len(use):
            n1 = int((use["label"] == 1).sum())
            print(f"  cycle_life  min {cl.min():.0f} / median {cl.median():.0f} / max {cl.max():.0f}")
            print(f"  label(>= {LABEL_THRESHOLD})  장수명 {n1} / 단수명 {len(use) - n1}"
                  f"  (장수명 비율 {n1 / len(use):.0%})")
            print(f"  policy_group  {use['policy_group'].value_counts().to_dict()}")
        corr = g[g["life_corrected"]]
        if len(corr):
            print(f"  수명 보정(+이어측정분): {', '.join(corr['cell_key'])}")

    out_rate = summary.groupby("batch")["qd_outlier"].mean()
    print("\n  QD 이상치 비율(행 기준): " + ", ".join(f"B{b} {r:.2%}" for b, r in out_rate.items()))

    use = cells[~cells["exclude"] & ~cells["curve_truncated"]]
    diff = (use["cycle_life_observed"] - use["cycle_life"]).dropna()
    if len(diff):
        print(f"  cycle_life vs 관측 EOL(QD<{EOL_AH:.2f}) 차이: median {diff.median():+.0f}, "
              f"|diff|>50 인 셀 {int((diff.abs() > 50).sum())}개")


# ---------------------------------------------------------------------------
# 캐시 로더
# ---------------------------------------------------------------------------
def load_processed(processed_dir: Path | str = PROCESSED_DIR, include_excluded: bool = False):
    """
    returns
    -------
    cells   : pd.DataFrame (셀 단위)
    summary : pd.DataFrame (사이클 단위)
    qd      : dict  {'cell_keys', 'cycles', 'Qdlin', 'Tdlin', 'V', 'index'}
              Qdlin/Tdlin shape = (n_cells, max_cycle+1, 1000), index = {cell_key: row}
    """
    d = Path(processed_dir)
    if not (d / "cells.parquet").exists():
        raise FileNotFoundError(f"{d} 에 캐시가 없습니다. 먼저 `python -m src.preprocess` 를 실행하세요.")

    cells = pd.read_parquet(d / "cells.parquet")
    summary = pd.read_parquet(d / "summary.parquet")
    z = np.load(d / "qdlin.npz", allow_pickle=False)
    keys = z["cell_keys"].astype(str)
    Q, T = z["Qdlin"], z["Tdlin"]

    if not include_excluded:
        cells = cells[~cells["exclude"]].reset_index(drop=True)
        keep = set(cells["cell_key"])
        summary = summary[summary["cell_key"].isin(keep)].reset_index(drop=True)
        rows = np.array([k in keep for k in keys])
        keys, Q, T = keys[rows], Q[rows], T[rows]

    qd = {
        "cell_keys": keys,
        "cycles": z["cycles"],
        "Qdlin": Q,
        "Tdlin": T,
        "V": z["V"],
        "index": {k: n for n, k in enumerate(keys)},
    }
    return cells, summary, qd


def get_curve(qd: dict, cell_key: str, cycle_idx: int, kind: str = "Qdlin") -> np.ndarray:
    """특정 셀/사이클의 Qdlin(또는 Tdlin) 곡선 (1000,)."""
    if cycle_idx > int(qd["cycles"].max()):
        raise ValueError(f"cycle_idx {cycle_idx} 는 저장 범위(0~{qd['cycles'].max()}) 밖입니다. "
                         f"--max-cycle 을 늘려 다시 전처리하세요.")
    return qd[kind][qd["index"][cell_key], cycle_idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="MIT-Stanford 배터리 .mat → data/processed 캐시")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    ap.add_argument("--max-cycle", type=int, default=DEFAULT_MAX_CYCLE,
                    help="Qdlin/Tdlin 을 저장할 최대 cycle_idx (기본 100)")
    ap.add_argument("--no-discharge-time", action="store_true",
                    help="사이클 시계열로 방전시간을 계산하지 않음 (더 빠름)")
    args = ap.parse_args(argv)
    run(args.raw_dir, args.out_dir, args.max_cycle, not args.no_discharge_time)


if __name__ == "__main__":
    sys.exit(main())
