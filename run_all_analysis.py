from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# statsmodels (LMM)
import statsmodels.formula.api as smf

# ========= Poster-style colors =========
POSTER_BLUE = "#86ADD3"
POSTER_NAVY = "#2F4F90"
POSTER_LIGHT = "#DCE8F5"
POSTER_CREAM = "#EADFB8"
POSTER_SLATE = "#5F7FA9"
POSTER_MIST = "#B8CCE3"

# ========= Helpers =========
def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)

def pick_input_csv(folder: Path) -> Path:
    # 优先选包含 allcrews / lmm / merged 字样的
    cands = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No CSV found in {folder}")

    score = []
    for p in cands:
        name = p.name.lower()
        s = 0
        for kw in ["qc", "all", "crew", "allcrews", "lmm", "ready", "merged", "physio", "tlx", "rpsa"]:
            if kw in name:
                s += 1
        score.append((s, p))
    score.sort(key=lambda x: (x[0], x[1].stat().st_mtime), reverse=True)
    return score[0][1]


def fit_mixedlm(data: pd.DataFrame, predictor: str):
    sub = data.dropna(subset=[predictor, "RPSA_mean", "logRMSSD", "Crew", "Seat"]).copy()
    if sub.empty:
        raise ValueError(f"No usable rows for predictor {predictor}")

    sub[f"z_{predictor}"] = zscore(sub[predictor])
    sub["z_RPSA"] = zscore(sub["RPSA_mean"])
    sub["Crew_str"] = sub["Crew"].astype(int).astype(str)

    model = smf.mixedlm(
        f"logRMSSD ~ z_{predictor} * z_RPSA + Seat",
        data=sub,
        groups=sub["Crew_str"]
    )
    res = model.fit(method="lbfgs")
    out = pd.DataFrame({
        "predictor": predictor,
        "term": res.params.index,
        "estimate": res.params.values,
        "std_err": res.bse.values,
        "z": (res.params.values / res.bse.values),
        "p_value": res.pvalues.values,
        "n_obs": len(sub),
        "n_crews": sub["Crew"].nunique(),
    })
    return res, out, sub


def fit_random_slope_model(data: pd.DataFrame, predictor: str):
    sub = data.dropna(subset=[predictor, "logRMSSD", "Crew", "Seat"]).copy()
    if sub.empty:
        raise ValueError(f"No usable rows for predictor {predictor}")

    sub[f"z_{predictor}"] = zscore(sub[predictor])
    sub["Crew_str"] = sub["Crew"].astype(int).astype(str)

    model = smf.mixedlm(
        f"logRMSSD ~ z_{predictor} + Seat",
        data=sub,
        groups=sub["Crew_str"],
        re_formula=f"~z_{predictor}",
    )
    res = model.fit(method="lbfgs")
    return res, sub


def compute_crew_specific_slopes(data: pd.DataFrame, predictor: str) -> pd.DataFrame:
    rows = []
    for crew, sub in data.groupby("Crew"):
        sub = sub.dropna(subset=[predictor, "logRMSSD"]).copy()
        n_obs = len(sub)
        if n_obs < 2 or sub[predictor].nunique() < 2:
            slope = np.nan
            intercept = np.nan
        else:
            slope, intercept = np.polyfit(sub[predictor], sub["logRMSSD"], 1)

        rows.append({
            "Crew": int(crew),
            "n_obs": n_obs,
            "tlx_min": sub[predictor].min(),
            "tlx_max": sub[predictor].max(),
            "mean_logRMSSD": sub["logRMSSD"].mean(),
            "slope": slope,
            "intercept": intercept,
        })

    slopes = pd.DataFrame(rows).sort_values("slope", na_position="last").reset_index(drop=True)
    slopes["slope_direction"] = np.select(
        [slopes["slope"] > 0.002, slopes["slope"] < -0.002],
        ["positive", "negative"],
        default="near-flat",
    )
    return slopes

# ========= Paths =========
OUT_DIR = Path(os.environ.get("SOTERIA_OUT_DIR", "results")).expanduser()
RES_DIR = OUT_DIR / "analysis"
FIG_DIR = RES_DIR / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(os.environ.get("SOTERIA_INPUT_CSV", str(OUT_DIR / "soteria_allcrews_physio_tlx_rpsa_qc.csv"))).expanduser()
print(f"✅ Using input: {DATA_PATH}")

# ========= Load =========
df = pd.read_csv(DATA_PATH)

# ========= Minimal column checks =========
# 你现在数据里大概率有这些列（名字可能略不同）
needed_any = ["Crew", "Seat", "TLX_raw", "RPSA_mean"]
missing = [c for c in needed_any if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns {missing}. Columns present: {list(df.columns)}")

# HRV: 优先用 logRMSSD；没有就用 RMSSD_ms 算
if "logRMSSD" not in df.columns:
    if "RMSSD_ms" not in df.columns:
        raise ValueError("Need either logRMSSD or RMSSD_ms in your CSV.")
    df["RMSSD_ms"] = pd.to_numeric(df["RMSSD_ms"], errors="coerce")
    # 避免 log(0)
    df["logRMSSD"] = np.log(df["RMSSD_ms"].clip(lower=1e-6))

# types
df["Crew"] = pd.to_numeric(df["Crew"], errors="coerce").astype("Int64")
df["Seat"] = df["Seat"].astype(str).str.lower().astype("category")
df["TLX_raw"] = pd.to_numeric(df["TLX_raw"], errors="coerce")
df["RPSA_mean"] = pd.to_numeric(df["RPSA_mean"], errors="coerce")

for col in ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "QC_Pass_HRV" in df.columns:
    df["QC_Pass_HRV"] = df["QC_Pass_HRV"].astype(str).str.lower().map({"true": True, "false": False})
    qc_mask = df["QC_Pass_HRV"].fillna(False)
else:
    qc_mask = (
        df["N_beats"].ge(30)
        & df["Duration_s"].ge(300)
        & df["logRMSSD"].notna()
    )
    df["QC_Pass_HRV"] = qc_mask

# drop rows with critical missing after HRV QC
df0 = df.loc[qc_mask].dropna(subset=["Crew", "Seat", "TLX_raw", "RPSA_mean", "logRMSSD"]).copy()

print("✅ N rows (after dropna):", len(df0))
print("✅ Crews:", df0["Crew"].nunique())

# ========= Descriptives =========
desc_cols = ["TLX_raw", "RPSA_mean", "logRMSSD", "N_beats", "Duration_s"]
desc = df0[desc_cols].describe().T
desc.to_csv(RES_DIR / "descriptives.csv")
print("✅ Wrote descriptives.csv")

# ========= Plot 1: TLX vs logRMSSD, colored by RPSA (median split) =========
df0["RPSA_group"] = np.where(df0["RPSA_mean"] >= df0["RPSA_mean"].median(),
                             "High RPSA", "Low RPSA")

plt.figure(figsize=(6.2, 4.8))
group_colors = {
    "Low RPSA": POSTER_MIST,
    "High RPSA": POSTER_CREAM,
}
for grp in ["Low RPSA", "High RPSA"]:
    sub = df0[df0["RPSA_group"] == grp]
    plt.scatter(
        sub["TLX_raw"], sub["logRMSSD"],
        s=34, alpha=0.8, label=grp,
        color=group_colors[grp], edgecolor=POSTER_NAVY, linewidth=0.5
    )

# overall descriptive trend line
if df0["TLX_raw"].nunique() > 1:
    slope, intercept = np.polyfit(df0["TLX_raw"], df0["logRMSSD"], 1)
    xline = np.linspace(df0["TLX_raw"].min(), df0["TLX_raw"].max(), 100)
    yline = intercept + slope * xline
    plt.plot(xline, yline, color=POSTER_NAVY, linewidth=2.8, alpha=0.95)

plt.xlabel("NASA-TLX (raw)")
plt.ylabel("log(RMSSD)")
plt.title("TLX vs log(RMSSD) (colored by RPSA group)")
plt.legend(frameon=True, facecolor="white", edgecolor=POSTER_SLATE)
plt.tight_layout()
plt.savefig(FIG_DIR / "scatter_tlx_logrmssd_rpsa_groups.png", dpi=300)
plt.close()

# ========= Plot 2: within-crew line (spaghetti) =========
plt.figure(figsize=(6.2, 4.8))
crew_colors = [
    POSTER_BLUE, POSTER_NAVY, POSTER_CREAM, POSTER_SLATE,
    "#A8C3E0", "#6D8FB7", "#CDBE91", "#9FB6CF"
]
for idx, (crew, sub) in enumerate(df0.groupby("Crew")):
    sub = sub.sort_values("TLX_raw")
    plt.plot(
        sub["TLX_raw"], sub["logRMSSD"],
        alpha=0.65, linewidth=2.4,
        color=crew_colors[idx % len(crew_colors)]
    )

plt.xlabel("NASA-TLX (raw)")
plt.ylabel("log(RMSSD)")
plt.title("Within-crew pattern (spaghetti)")
plt.tight_layout()
plt.savefig(FIG_DIR / "spaghetti_within_crew.png", dpi=300)
plt.close()

print("✅ Saved figures to:", FIG_DIR)

# ========= Exploratory random-slope follow-up =========
try:
    rs_res, rs_data = fit_random_slope_model(df0, "TLX_raw")
    (RES_DIR / "model_summary_tlx_random_slope.txt").write_text(rs_res.summary().as_text(), encoding="utf-8")
    print("✅ Wrote model_summary_tlx_random_slope.txt")

    rand_eff = []
    for crew, eff in rs_res.random_effects.items():
        rand_eff.append({
            "Crew": int(crew),
            "random_intercept": float(eff.get("Group", np.nan)),
            "random_slope_z_TLX_raw": float(eff.get("z_TLX_raw", np.nan)),
        })
    rand_eff_df = pd.DataFrame(rand_eff).sort_values("Crew").reset_index(drop=True)
    rand_eff_df.to_csv(RES_DIR / "random_effects_tlx_random_slope.csv", index=False)
    print("✅ Wrote random_effects_tlx_random_slope.csv")
except Exception as e:
    rs_res = None
    rs_data = None
    print(f"⚠️ Random-slope model failed: {e}")

crew_slopes = compute_crew_specific_slopes(df0, "TLX_raw")
crew_slopes.to_csv(RES_DIR / "crew_specific_tlx_slopes.csv", index=False)
print("✅ Wrote crew_specific_tlx_slopes.csv")

plot_slopes = crew_slopes.dropna(subset=["slope"]).copy()
if not plot_slopes.empty:
    colors = plot_slopes["slope"].map(lambda x: POSTER_NAVY if x > 0 else POSTER_BLUE)
    labels = [f"C{int(crew):02d}\n(n={int(n_obs)})" for crew, n_obs in zip(plot_slopes["Crew"], plot_slopes["n_obs"])]
    plt.figure(figsize=(7.5, 4.8))
    plt.bar(labels, plot_slopes["slope"], color=colors, edgecolor=POSTER_NAVY, linewidth=0.8)
    plt.axhline(0, color=POSTER_NAVY, linewidth=1.4)
    plt.xlabel("Crew")
    plt.ylabel("Slope: TLX -> log(RMSSD)")
    plt.title("Crew-specific workload-HRV slopes")
    plt.figtext(
        0.5, -0.04,
        "Exploratory descriptive slopes; some crews contribute few observations.",
        ha="center", fontsize=9
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "crew_specific_tlx_slopes.png", dpi=300)
    plt.close()
    print("✅ Wrote crew_specific_tlx_slopes.png")

    # Lollipop version for poster comparison
    y_pos = np.arange(len(plot_slopes))
    plt.figure(figsize=(7.5, 4.8))
    for y, slope, color in zip(y_pos, plot_slopes["slope"], colors):
        plt.hlines(y=y, xmin=0, xmax=slope, color=color, linewidth=3)
        plt.scatter(slope, y, s=90, color=color, edgecolor=POSTER_NAVY, linewidth=0.8, zorder=3)

    plt.axvline(0, color=POSTER_NAVY, linewidth=1.4)
    plt.yticks(y_pos, labels)
    plt.xlabel("Slope: TLX -> log(RMSSD)")
    plt.ylabel("Crew")
    plt.title("Crew-specific workload-HRV slopes")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", color=POSTER_LIGHT, linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "crew_specific_tlx_slopes_lollipop.png", dpi=300)
    plt.close()
    print("✅ Wrote crew_specific_tlx_slopes_lollipop.png")

# ========= LMM =========
predictors = ["TLX_raw"]
predictors.extend([c for c in ["Mental", "Physical", "Temporal", "Performance", "Effort", "Frustration"] if c in df0.columns])

all_fe = []
for predictor in predictors:
    try:
        res, fe, _ = fit_mixedlm(df0, predictor)
    except Exception as e:
        print(f"⚠️ Model failed for {predictor}: {e}")
        continue

    summary_path = RES_DIR / f"model_summary_{predictor.lower()}.txt"
    summary_path.write_text(res.summary().as_text(), encoding="utf-8")
    all_fe.append(fe)
    print(f"✅ Wrote {summary_path.name}")

if not all_fe:
    raise RuntimeError("No models converged. Check the QC-filtered dataset.")

fe_all = pd.concat(all_fe, ignore_index=True)
fe_all.to_csv(RES_DIR / "fixed_effects_all_models.csv", index=False)
fe_all.loc[fe_all["predictor"] == "TLX_raw"].to_csv(RES_DIR / "fixed_effects.csv", index=False)
print("✅ Wrote fixed_effects_all_models.csv")
print("✅ Wrote fixed_effects.csv")

key_terms = fe_all[
    fe_all["term"].isin(["z_TLX_raw", "z_Mental", "z_Physical", "z_Temporal", "z_Performance", "z_Effort", "z_Frustration"])
].sort_values("p_value")

print("\n===== TLX MAIN EFFECTS =====")
for _, row in key_terms.iterrows():
    print(
        f"{row['predictor']:12s}  estimate={row['estimate']:+.4f}  "
        f"SE={row['std_err']:.4f}  z={row['z']:+.2f}  p={row['p_value']:.3f}"
    )

if not crew_slopes.empty:
    print("\n===== CREW-SPECIFIC TLX SLOPES =====")
    for _, row in crew_slopes.iterrows():
        if pd.notna(row["slope"]):
            print(
                f"Crew {int(row['Crew']):02d}  n={int(row['n_obs'])}  "
                f"slope={row['slope']:+.4f}  ({row['slope_direction']})"
            )

print("\n✅ DONE. Open:", RES_DIR)
