# Soteria APA poster analysis

Python scripts for the Soteria workload, RPSA, and heart-rate variability analysis prepared by Xinzhu Chai for sharing with collaborators.

## Scope and provenance

This repository packages the locally available `build_all.py` and `run_all_analysis.py`. Statistical formulas, preprocessing, QC thresholds, and plotting logic are preserved. Packaging changes replace personal filesystem paths with environment variables and select the QC dataset explicitly instead of guessing among CSV files. The final APA poster was not available for comparison: this snapshot has not been certified as the exact poster submission version. The analysis script also contains exploratory follow-up analyses.

Original source SHA-256 hashes are recorded in `source_manifest.json`. Earlier standalone/experimental scripts, raw data, questionnaire workbooks, and participant-level results are not included.

## Setup

Use Python 3.10 or newer. From this repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Dependency ranges are compatibility targets, not a lockfile of the historical analysis environment.

## Required data

Obtain the Soteria data and survey workbook separately through your existing authorized data source. This repository does not download or redistribute them.

Default layout:

```text
data/
  soteria-survey-data.xlsx
  soteria_data/
    Crew_01/
      Processing/
        emp_ibi_leftseat_scenario7.csv
        ...
    Crew_02/
      Processing/
        ...
```

The builder checks Crew_01 through Crew_13 and reports missing or skipped crews. IBI CSVs require a timestamp column (`UserTimeStamp`, `user_time_stamp`, `timestamp`, or `time`) in seconds and an interval column (`IBI`, `ibi_s`, `ibi_sec`, or `interbeatinterval`) in seconds. Files must follow the seat/scenario naming pattern above. The original pipeline retains input row order for consecutive-interval calculations; supply chronologically ordered files.

The workbook requires:

- `TLX` sheet: Crew, Seat, Scenario, Mental, Physical, Temporal, Performance, Effort, Frustration.
- `RPSA` sheet: Crew, Seat, and numerically named item columns (1, 2, ...).
- Seat values matching `left` / `right` for joins.

## Run

```bash
python build_all.py
MPLBACKEND=Agg python run_all_analysis.py
```

For data stored elsewhere, set these environment variables (macOS/Linux):

```bash
export SOTERIA_DATA_ROOT="/path/to/soteria_data"
export SOTERIA_SURVEY_XLSX="/path/to/soteria-survey-data.xlsx"
export SOTERIA_OUT_DIR="results"
python build_all.py
MPLBACKEND=Agg python run_all_analysis.py
```

To analyze an already merged table:

```bash
SOTERIA_INPUT_CSV="/path/to/merged_qc.csv" MPLBACKEND=Agg python run_all_analysis.py
```

The merged input needs Crew, Seat, TLX_raw, RPSA_mean, N_beats, Duration_s, and logRMSSD (or RMSSD_ms). QC_Pass_HRV is used when present; otherwise the analysis applies its fallback beat-count/duration checks. Optional TLX dimension columns enable the corresponding follow-up models.

## Analysis implemented

1. Retain IBI values between 0.3 and 2.0 seconds; compute RMSSD, SDNN, and mean heart rate per Crew × Seat × Scenario.
2. Apply the existing QC rule: at least 30 retained beats, at least 300 seconds timestamp span, and nonmissing RMSSD/SDNN. This is the historical rule, not a complete artifact-correction procedure.
3. Average available TLX dimensions and available RPSA items, then merge with physiological summaries. Missing items are skipped in these means; no additional reverse-scoring step is implemented.
4. Fit REML mixed models: `logRMSSD ~ z_predictor * z_RPSA + Seat`, with a random intercept for Crew. Predictors include overall TLX and each available dimension. Standardization uses population SD (ddof=0) within the relevant analysis subset.
5. Generate RPSA median-split scatter and within-crew plots. The median split is for visualization; models use continuous RPSA.
6. Fit an exploratory TLX random-slope model and calculate descriptive within-crew slopes. Descriptive slopes are separate from mixed-model random slopes.

P-values are unadjusted. The original export named `fixed_effects*.csv` includes the variance parameter as well as fixed effects. Model fitting warnings and convergence flags need review: the script catches some model failures and may continue, so a zero exit code alone does not establish convergence of every model. A nonempty fit collection also does not guarantee convergence, despite the historical error message wording.

## Outputs

`results/` contains full and QC-filtered merged tables. `results/analysis/` contains model summaries, parameter tables, descriptive statistics, crew slopes, and a `figures/` subdirectory. Repeated runs overwrite files with the same names; use a fresh output directory when comparing runs. Generated outputs and data are excluded from Git.

## Verification status

The packaged scripts passed Python syntax parsing. A complete rerun was not performed during packaging because the available Python environment lacked statsmodels, scipy, and openpyxl. The historical saved overall-TLX model reported 40 observations across 8 crews and convergence; this is a reference from saved output, not a new reproduction claim.
