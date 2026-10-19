import re
from pathlib import Path
import os
import numpy as np
import pandas as pd


# ========= 你需要改的地方（通常只改 DATA_ROOT 和 SURVEY_XLSX） =========
# 你下载 S3 数据的本地根目录（建议你把所有 Crew_XX 放在这个目录下）
# 例如：~/soteria_data/Crew_01/Processing/...
DATA_ROOT = Path(os.environ.get("SOTERIA_DATA_ROOT", "data/soteria_data")).expanduser()

# 问卷 excel（TLX/RPSA）文件位置
SURVEY_XLSX = Path(os.environ.get("SOTERIA_SURVEY_XLSX", "data/soteria-survey-data.xlsx")).expanduser()

# 输出位置
OUT_DIR = Path(os.environ.get("SOTERIA_OUT_DIR", "results")).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "soteria_allcrews_physio_tlx_rpsa.csv"
OUT_QC_CSV = OUT_DIR / "soteria_allcrews_physio_tlx_rpsa_qc.csv"

# HRV 片段的最基本质量门槛：至少约 5 分钟、且有足够 beats
MIN_HRV_BEATS = 30
MIN_HRV_DURATION_S = 300.0
# ======================================================================


def parse_seat_scenario(filename: str):
    """
    emp_ibi_leftseat_scenario7.csv -> seat=left, scenario=7
    """
    seat = None
    scenario = None

    m_seat = re.search(r"(leftseat|rightseat)", filename, re.IGNORECASE)
    if m_seat:
        seat = "left" if "left" in m_seat.group(1).lower() else "right"

    m_scn = re.search(r"scenario(\d+)", filename, re.IGNORECASE)
    if m_scn:
        scenario = int(m_scn.group(1))

    return seat, scenario


def rmssd(rr_ms: np.ndarray) -> float:
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    if rr_ms.size < 3:
        return np.nan
    diff = np.diff(rr_ms)
    return float(np.sqrt(np.mean(diff ** 2)))


def sdnn(rr_ms: np.ndarray) -> float:
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    if rr_ms.size < 3:
        return np.nan
    return float(np.std(rr_ms, ddof=1))


def mean_hr(rr_ms: np.ndarray) -> float:
    rr_ms = rr_ms[np.isfinite(rr_ms)]
    if rr_ms.size < 1:
        return np.nan
    return float(np.mean(60000.0 / rr_ms))


def read_ibi_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp)

    # 去掉空列 / Unnamed 列
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", na=False)]

    # 标准化列名
    col_map = {}
    for c in df.columns:
        c2 = str(c).strip().lower()
        if c2 in ["usertimestamp", "user_time_stamp", "timestamp", "time"]:
            col_map[c] = "UserTimeStamp"
        elif c2 in ["ibi", "ibi_s", "ibi_sec", "interbeatinterval"]:
            col_map[c] = "IBI"
    df = df.rename(columns=col_map)

    if "UserTimeStamp" not in df.columns or "IBI" not in df.columns:
        raise ValueError(f"{fp.name} 缺少必要列。当前列名：{list(df.columns)}")

    df["UserTimeStamp"] = pd.to_numeric(df["UserTimeStamp"], errors="coerce")
    df["IBI"] = pd.to_numeric(df["IBI"], errors="coerce")
    df = df.dropna(subset=["UserTimeStamp", "IBI"]).copy()

    # IBI 合理范围（秒）
    df = df[(df["IBI"] >= 0.3) & (df["IBI"] <= 2.0)].copy()

    return df


def build_physio_summary_for_crew(crew_id: int, processing_dir: Path) -> pd.DataFrame:
    """
    从某个 crew 的 Processing 目录里读取所有 emp_ibi_*scenario*.csv
    -> 拼成长表 -> 汇总成每个 Crew×Seat×Scenario 一行 HRV 指标
    """
    files = sorted(processing_dir.glob("emp_ibi_*scenario*.csv"))
    if not files:
        raise FileNotFoundError(f"{processing_dir} 找不到 emp_ibi_*scenario*.csv")

    all_rows = []
    for fp in files:
        seat, scenario = parse_seat_scenario(fp.name)
        if seat is None or scenario is None:
            continue

        df = read_ibi_csv(fp)

        df.insert(0, "Crew", crew_id)
        df.insert(1, "Seat", seat)
        df.insert(2, "Scenario", scenario)
        df.insert(3, "SourceFile", fp.name)

        df["IBI_ms"] = df["IBI"] * 1000.0
        all_rows.append(df[["Crew", "Seat", "Scenario", "SourceFile", "UserTimeStamp", "IBI", "IBI_ms"]])

    if not all_rows:
        raise ValueError(f"{processing_dir} 里没有可用的 IBI 文件（seat/scenario 解析失败）")

    ibi_long = pd.concat(all_rows, ignore_index=True)

    # 汇总成 HRV 指标（每个 Crew×Seat×Scenario 一行）
    summary = (
        ibi_long.groupby(["Crew", "Seat", "Scenario"], as_index=False)
        .agg(
            N_beats=("IBI_ms", "size"),
            Duration_s=("UserTimeStamp", lambda x: float(np.nanmax(x) - np.nanmin(x)) if len(x) else np.nan),
            Mean_HR_bpm=("IBI_ms", lambda x: mean_hr(x.to_numpy())),
            RMSSD_ms=("IBI_ms", lambda x: rmssd(x.to_numpy())),
            SDNN_ms=("IBI_ms", lambda x: sdnn(x.to_numpy())),
        )
    )

    summary["QC_MinBeats"] = summary["N_beats"] >= MIN_HRV_BEATS
    summary["QC_MinDuration"] = summary["Duration_s"] >= MIN_HRV_DURATION_S
    summary["QC_Pass_HRV"] = (
        summary["QC_MinBeats"]
        & summary["QC_MinDuration"]
        & summary["RMSSD_ms"].notna()
        & summary["SDNN_ms"].notna()
    )

    # 不通过 QC 的片段保留元数据，但清空 HRV 指标，避免后续分析误用
    invalid_mask = ~summary["QC_Pass_HRV"]
    summary.loc[invalid_mask, ["Mean_HR_bpm", "RMSSD_ms", "SDNN_ms"]] = np.nan
    return summary


def load_tlx() -> pd.DataFrame:
    tlx = pd.read_excel(SURVEY_XLSX, sheet_name="TLX")
    tlx.columns = [str(c).strip() for c in tlx.columns]

    # 必要列
    crew_col = next((c for c in tlx.columns if c.lower() == "crew"), None)
    seat_col = next((c for c in tlx.columns if c.lower() == "seat"), None)
    scn_col = next((c for c in tlx.columns if c.lower() in ["scenario", "scenario id", "scenario_id"]), None)
    if not (crew_col and seat_col and scn_col):
        raise ValueError(f"TLX 缺少 Crew/Seat/Scenario 列。当前列：{tlx.columns.tolist()}")

    dims = ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]
    for d in dims:
        if d not in tlx.columns:
            raise ValueError(f"TLX 缺少维度列 {d}。当前列：{tlx.columns.tolist()}")

    tlx = tlx.rename(columns={crew_col: "Crew", seat_col: "Seat", scn_col: "Scenario"})
    tlx["Crew"] = pd.to_numeric(tlx["Crew"], errors="coerce").astype("Int64")
    tlx["Scenario"] = pd.to_numeric(tlx["Scenario"], errors="coerce").astype("Int64")

    for d in dims:
        tlx[d] = pd.to_numeric(tlx[d], errors="coerce")

    tlx["TLX_raw"] = tlx[dims].mean(axis=1, skipna=True)

    # 如果同一 Crew×Seat×Scenario 有多行 TLX，这里聚合成一行（mean）
    tlx_summary = (
        tlx.groupby(["Crew", "Seat", "Scenario"], as_index=False)
        .agg(
            TLX_raw=("TLX_raw", "mean"),
            Mental=("Mental", "mean"),
            Physical=("Physical", "mean"),
            Temporal=("Temporal", "mean"),
            Performance=("Performance", "mean"),
            Effort=("Effort", "mean"),
            Frustration=("Frustration", "mean"),
        )
    )
    return tlx_summary


def load_rpsa() -> pd.DataFrame:
    rpsa = pd.read_excel(SURVEY_XLSX, sheet_name="RPSA")
    rpsa.columns = [str(c).strip() for c in rpsa.columns]

    crew_col = next((c for c in rpsa.columns if c.lower() == "crew"), None)
    seat_col = next((c for c in rpsa.columns if c.lower() == "seat"), None)
    if not (crew_col and seat_col):
        raise ValueError(f"RPSA 缺少 Crew/Seat 列。当前列：{rpsa.columns.tolist()}")

    item_cols = [c for c in rpsa.columns if str(c).isdigit()]
    if not item_cols:
        raise ValueError(f"RPSA 没找到题号列（1..）。当前列：{rpsa.columns.tolist()}")

    rpsa = rpsa.rename(columns={crew_col: "Crew", seat_col: "Seat"})
    rpsa["Crew"] = pd.to_numeric(rpsa["Crew"], errors="coerce").astype("Int64")
    for c in item_cols:
        rpsa[c] = pd.to_numeric(rpsa[c], errors="coerce")

    rpsa["RPSA_mean"] = rpsa[item_cols].mean(axis=1, skipna=True)

    # 如果同一 Crew×Seat 有多行 RPSA，这里聚合成一行（mean）
    rpsa_summary = (
        rpsa.groupby(["Crew", "Seat"], as_index=False)
        .agg(RPSA_mean=("RPSA_mean", "mean"))
    )
    return rpsa_summary


def main():
    if not SURVEY_XLSX.exists():
        raise FileNotFoundError(f"找不到 SURVEY_XLSX: {SURVEY_XLSX}\n请把路径改对。")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"找不到 DATA_ROOT: {DATA_ROOT}\n请把路径改成你 Crew_XX 存放的根目录。")

    tlx = load_tlx()
    rpsa = load_rpsa()

    all_crew_tables = []
    skipped = []

    for crew_id in range(1, 14):  # 01-13
        crew_name = f"Crew_{crew_id:02d}"
        processing_dir = DATA_ROOT / crew_name / "Processing"
        if not processing_dir.exists():
            skipped.append((crew_name, "missing Processing dir"))
            continue

        try:
            phys = build_physio_summary_for_crew(crew_id, processing_dir)
            merged = phys.merge(tlx, on=["Crew", "Seat", "Scenario"], how="left")
            merged = merged.merge(rpsa, on=["Crew", "Seat"], how="left")
            all_crew_tables.append(merged)
            print(f"✅ {crew_name}: physio rows={len(phys)} merged rows={len(merged)}")
        except Exception as e:
            skipped.append((crew_name, str(e)))
            print(f"⚠️ {crew_name} skipped: {e}")

    if not all_crew_tables:
        raise RuntimeError("没有任何 crew 成功生成数据。请检查 DATA_ROOT 结构与文件是否存在。")

    master = pd.concat(all_crew_tables, ignore_index=True)

    # 排序让输出更好看
    master = master.sort_values(["Crew", "Seat", "Scenario"]).reset_index(drop=True)
    master["logRMSSD"] = np.where(master["QC_Pass_HRV"], np.log(master["RMSSD_ms"]), np.nan)

    qc_master = master.loc[master["QC_Pass_HRV"]].copy()

    master.to_csv(OUT_CSV, index=False)
    qc_master.to_csv(OUT_QC_CSV, index=False)
    print(f"\n🎉 DONE. Saved master dataset -> {OUT_CSV}")
    print(f"✅ Saved QC-passed dataset -> {OUT_QC_CSV}")
    print("Master shape:", master.shape)
    print("QC-passed shape:", qc_master.shape)

    if skipped:
        print("\n--- Skipped crews ---")
        for crew_name, reason in skipped:
            print(f"{crew_name}: {reason}")


if __name__ == "__main__":
    main()
