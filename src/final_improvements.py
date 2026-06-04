"""
final_improvements.py — Phase 0 quality improvements pipeline.

Implements all code-side Phase 0 improvements to the original paper
'Regime-Conditioned Trade Flow Imbalance and Adverse Selection in ES
Futures' (Ungab, 2026). All improvements run on the in-sample trades
dataset only. The MBO external drive is NOT accessed in this script.

PLACEMENT: This script belongs in src/ alongside formal_analysis.py,
data_loader.py, and signal_construction.py in the es-regime-tfi-trades
repository. RESULTS_DIR resolves to results/final-improvements/ relative
to the script's location (i.e., one level up from src/).

Phase 0 code items:
    P0-1  R² Decomposition                         [IMPLEMENTED]
    P0-2  Threshold Sensitivity                    [IMPLEMENTED]
    P0-3  Additive Combination Robustness          [PENDING]
    P0-4  Lambda and TAR Window Sensitivity        [PENDING]
    P0-5  Expanded Announcement Exclusion Set      [PENDING]
    P0-6  Pre-Announcement Window Characterization [PENDING]
    P0-7  Formal Bias Simulation                   [PENDING]

Writing revisions P0-8, P0-9, P0-10 are applied directly to PAPER.md
and are not implemented here.

Outputs written to results/final-improvements/:
    p0_1_primary_no_lag_return.txt
    p0_2_threshold_sensitivity.txt
    p0_3_additive_regression.txt           [PENDING]
    p0_4_window_sensitivity.csv            [PENDING]
    p0_5_expanded_exclusion.txt            [PENDING]
    p0_6_preannouncement_stats.txt         [PENDING]
    p0_7_simulation_full.txt               [PENDING]
    p0_7_simulation_stable.txt             [PENDING]
    p0_7_simulation_histogram.png          [PENDING]
    p0_key_results.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- Path setup --------------------------------------------------------------
# Resolves imports of data_loader and signal_construction from the same
# directory as this script, regardless of working directory at runtime.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from data_loader import load_all_days, remove_outliers, compute_tfi, compute_returns
from signal_construction import (
    compute_lambda,
    compute_arrival_rate,
    compute_exclusion_mask,
    compute_regime_score,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = os.path.expanduser(
    '/Volumes/X9 Pro/raw-market-data/es-futures/trades/GLBX-20250501-20251231/'
)

# Resolves to results/final-improvements/ in the repo root when this
# script is placed in src/.
RESULTS_DIR = os.path.join(_SCRIPT_DIR, '..', 'results', 'final-improvements')
os.makedirs(RESULTS_DIR, exist_ok=True)

TZ       = 'America/New_York'
HAC_LAGS = 5

# Scheduled macro announcement datetimes (Eastern) — in-sample period only.
# Identical to ANNOUNCEMENT_DATES in formal_analysis.py.
ANNOUNCEMENT_DATES = [
    # FOMC decisions (2:00 PM ET)
    pd.Timestamp('2025-05-07 14:00', tz=TZ),
    pd.Timestamp('2025-06-18 14:00', tz=TZ),
    pd.Timestamp('2025-07-30 14:00', tz=TZ),
    pd.Timestamp('2025-09-17 14:00', tz=TZ),
    pd.Timestamp('2025-10-29 14:00', tz=TZ),
    pd.Timestamp('2025-12-10 14:00', tz=TZ),
    # CPI releases (8:30 AM ET)
    pd.Timestamp('2025-05-13 08:30', tz=TZ),
    pd.Timestamp('2025-06-11 08:30', tz=TZ),
    pd.Timestamp('2025-07-15 08:30', tz=TZ),
    pd.Timestamp('2025-08-12 08:30', tz=TZ),
    pd.Timestamp('2025-09-10 08:30', tz=TZ),
    pd.Timestamp('2025-12-18 08:30', tz=TZ),
    # NFP releases (8:30 AM ET)
    pd.Timestamp('2025-05-02 08:30', tz=TZ),
    pd.Timestamp('2025-06-06 08:30', tz=TZ),
    pd.Timestamp('2025-07-03 08:30', tz=TZ),
    pd.Timestamp('2025-08-01 08:30', tz=TZ),
    pd.Timestamp('2025-09-05 08:30', tz=TZ),
    pd.Timestamp('2025-11-20 08:30', tz=TZ),
    pd.Timestamp('2025-12-16 08:30', tz=TZ),
]

# Columns required to drop NaN rows once, covering warmup, day boundaries,
# and forward-return edge effects. Identical to formal_analysis.py.
REGRESSION_COLS = [
    'fwd_return', 'tfi', 'regime_score', 'tfi_x_regime',
    'regime_score_lag', 'tfi_x_regime_lag', 'lag_return', 'lag_tfi',
]

# Primary regression regressors (with mean-reversion control).
PRIMARY_COLS = ['tfi', 'regime_score', 'tfi_x_regime', 'lag_return', 'lag_tfi']

# Variables captured in p0_key_results.csv.
KEY_VARS = ['const', 'tfi', 'regime_score', 'tfi_x_regime', 'lag_return', 'lag_tfi']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# Identical to the implementations in formal_analysis.py.


def _fit_ols(y, X_cols, data):
    """Fit HAC-robust OLS and return the fitted model."""
    X = sm.add_constant(data[X_cols])
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})


def _print_coeff_table(model, variables):
    """Print a compact coefficient table for the specified variables."""
    print(f"  {'Variable':<25} {'Coeff':>12} {'z-stat':>8} {'p-value':>10}")
    print(f"  {'-' * 59}")
    for var in variables:
        if var not in model.params:
            continue
        c   = model.params[var]
        t   = model.tvalues[var]
        p   = model.pvalues[var]
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f"  {var:<25} {c:>12.6f} {t:>8.3f} {p:>10.4f} {sig}")
    print(f"  R² = {model.rsquared:.6f}  |  N = {int(model.nobs):,}")


def _save_model(model, filename):
    """Write a statsmodels summary to a text file in RESULTS_DIR."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        f.write(str(model.summary()))
    print(f"  Saved: {filename}")


def _collect_rows(fitted_model, model_label, rows):
    """
    Append coefficient rows for KEY_VARS from fitted_model to rows.
    Used to build the collated p0_key_results.csv output.
    """
    for var in KEY_VARS:
        if var not in fitted_model.params:
            continue
        rows.append({
            'model':     model_label,
            'variable':  var,
            'coeff':     fitted_model.params[var],
            't_stat':    fitted_model.tvalues[var],
            'p_value':   fitted_model.pvalues[var],
            'r_squared': fitted_model.rsquared,
            'n_obs':     int(fitted_model.nobs),
        })


def _build_reg_df(tfi_input, returns_input, regime_score_input):
    """
    Assemble the regression DataFrame from signal outputs, null out the
    first bar of each day (overnight return contamination), and construct
    all lagged and interaction terms.

    Identical to the implementation in formal_analysis.py.
    """
    tfi_s = tfi_input['tfi'] if isinstance(tfi_input, pd.DataFrame) else tfi_input
    ret_s = (returns_input['log_return']
             if isinstance(returns_input, pd.DataFrame) else returns_input)

    df = pd.DataFrame({
        'tfi':          tfi_s,
        'log_return':   ret_s,
        'regime_score': regime_score_input,
    })

    # Null the first bar of each trading day — the overnight gap makes
    # log_return at bar 0 an invalid RTH intraday return.
    dates = pd.Series(df.index.date, index=df.index)
    df.loc[dates != dates.shift(1), 'log_return'] = np.nan

    df['fwd_return']       = df['log_return'].shift(-1)
    df['lag_return']       = df['log_return']
    df['lag_tfi']          = df['tfi'].shift(1)
    df['tfi_x_regime']     = df['tfi'] * df['regime_score']
    df['regime_score_lag'] = df['regime_score'].shift(1)
    df['tfi_x_regime_lag'] = df['tfi'] * df['regime_score_lag']

    return df


# =============================================================================
# DATA LOADING AND SIGNAL CONSTRUCTION
# =============================================================================
# Loaded once at the top and shared across all P0 items.
# Pipeline is identical to formal_analysis.py sections 1-3.

print("=" * 60)
print("PHASE 0 — FINAL IMPROVEMENTS PIPELINE")
print("=" * 60)
print(f"\n  Results directory: {os.path.abspath(RESULTS_DIR)}")

print("\n[1] Loading in-sample data...")
df       = load_all_days(DATA_DIR)
df_clean = remove_outliers(df)
print(f"    {len(df_clean):,} clean RTH trades across "
      f"{df_clean['ts_event_et'].dt.date.nunique()} trading days")

print("\n[2] Computing in-sample signals...")
lambda_series  = compute_lambda(df_clean)
arrival_series = compute_arrival_rate(df_clean)

df_indexed = df_clean.set_index('ts_event_et')
bars       = df_indexed['price'].resample('1min').count()

ann_dates = ANNOUNCEMENT_DATES
if bars.index.tzinfo is None:
    ann_dates = [dt.tz_localize(None) for dt in ann_dates]

exclusion_mask = compute_exclusion_mask(bars, ann_dates)
regime_score   = compute_regime_score(lambda_series, arrival_series, exclusion_mask)
tfi            = compute_tfi(df_clean)
returns        = compute_returns(df_clean)

print("\n[3] Building regression DataFrame...")
reg     = _build_reg_df(tfi, returns, regime_score)
n_raw   = len(reg)
reg     = reg.dropna(subset=REGRESSION_COLS)
n_final = len(reg)
print(f"    Bars before filters:             {n_raw:,}")
print(f"    Dropped (NaN/warmup/boundaries): {n_raw - n_final:,}")
print(f"    Final regression N:              {n_final:,}")

# Collector for p0_key_results.csv — rows appended by each P0 section.
key_results_rows = []

# =============================================================================
# P0-1 — R² DECOMPOSITION
# =============================================================================
# Goal: clarify that R² = 0.236 in the primary regression is driven almost
# entirely by the mean-reversion control (lag_return), not by the regime-TFI
# interaction. Re-run the primary regression without lag_return and report
# both R² values side by side.
#
# Required outputs:
#   p0_1_primary_no_lag_return.txt   — full regression summary without Rₜ
#   p0_key_results.csv               — without-Rₜ regression rows appended

print("\n" + "=" * 60)
print("P0-1 — R² DECOMPOSITION")
print("=" * 60)

# --- With lag_return (sanity check — must reproduce original R² ≈ 0.236) ---
model_p01_with = _fit_ols(reg['fwd_return'], PRIMARY_COLS, reg)

# --- Without lag_return ---
_P01_COLS_NO_RT = ['tfi', 'regime_score', 'tfi_x_regime', 'lag_tfi']
model_p01_without = _fit_ols(reg['fwd_return'], _P01_COLS_NO_RT, reg)

r2_with    = model_p01_with.rsquared
r2_without = model_p01_without.rsquared

# Side-by-side R² comparison
print(f"\n  Primary regression R² comparison (N = {int(model_p01_with.nobs):,}):")
print(f"  {'Specification':<42} {'R²':>10}")
print(f"  {'-' * 54}")
print(f"  {'With lag_return (original spec)':<42} {r2_with:>10.6f}")
print(f"  {'Without lag_return':<42} {r2_without:>10.6f}")
print(f"\n  R² attributable to mean-reversion control:")
print(f"    Absolute drop: {r2_with - r2_without:.6f}")
print(f"    Share of total R²: {(r2_with - r2_without) / r2_with * 100:.1f}%")

# Sanity check
_EXPECTED_R2 = 0.236
if abs(r2_with - _EXPECTED_R2) < 0.001:
    print(f"\n  Sanity check: reproduced R² = {r2_with:.6f} "
          f"(expected ≈ {_EXPECTED_R2}) ✓")
else:
    print(f"\n  WARNING: reproduced R² = {r2_with:.6f} does not match "
          f"expected ≈ {_EXPECTED_R2}. Verify pipeline matches formal_analysis.py.")

# Coefficients for the without-control regression
print(f"\n  Without-control regression coefficients:")
_print_coeff_table(model_p01_without,
                   ['const', 'tfi', 'regime_score', 'tfi_x_regime', 'lag_tfi'])

# PAPER.md edit guidance printed to console for reference
print(f"\n  --- PAPER.md edit guidance (Section 5.1) ---")
print(f"  Append the following sentence immediately after the existing sentence")
print(f"  ending '...R² = 0.236 is driven almost entirely by this term.':")
print()
print(f"    Without the mean-reversion control, R² = {r2_without:.6f},")
print(f"    confirming that the regime-TFI interaction accounts for")
print(f"    essentially none of the return variance.")

# Save and collect
_save_model(model_p01_without, 'p0_1_primary_no_lag_return.txt')
_collect_rows(model_p01_without, 'p0_1_no_lag_return', key_results_rows)

# =============================================================================
# P0-2 — THRESHOLD SENSITIVITY
# =============================================================================
# Stress-test the high-regime threshold of 0.5 by re-running the detector
# validation regression (Equation 5) at thresholds 0.4, 0.5, and 0.6.
# Reports β₃ (within-bar TFI-return amplification), z-stat, p-value,
# amplification ratio, and high-regime bar fraction at each threshold.
# Threshold 0.5 row serves as a sanity check against the original result.
#
# Required outputs:
#   p0_2_threshold_sensitivity.txt   — comparison table + full summaries
#   p0_key_results.csv               — validation regression rows appended

print("\n" + "=" * 60)
print("P0-2 — THRESHOLD SENSITIVITY")
print("=" * 60)

_P02_THRESHOLDS   = [0.4, 0.5, 0.6]
_P02_VAL_VARS     = ['tfi', 'high_regime_dummy', 'tfi_x_high_regime', 'lag_tfi']
_P02_ORIG_B3      = 0.001525   # original β₃ from formal_analysis.py Section 3

p02_records = []

for thresh in _P02_THRESHOLDS:
    reg_val = reg.copy()
    reg_val['high_regime_dummy'] = (reg_val['regime_score'] > thresh).astype(float)
    reg_val['tfi_x_high_regime'] = reg_val['tfi'] * reg_val['high_regime_dummy']

    model_p02 = _fit_ols(reg_val['lag_return'], _P02_VAL_VARS, reg_val)

    b1  = model_p02.params['tfi']
    b3  = model_p02.params['tfi_x_high_regime']
    z3  = model_p02.tvalues['tfi_x_high_regime']
    p3  = model_p02.pvalues['tfi_x_high_regime']
    hi_frac = reg_val['high_regime_dummy'].mean()
    amp = (b1 + b3) / b1 if b1 != 0 else float('nan')

    p02_records.append({
        'thresh': thresh, 'hi_frac': hi_frac,
        'b3': b3, 'z3': z3, 'p3': p3, 'amp': amp,
        'model': model_p02,
    })

    # Collect all validation variables for p0_key_results.csv.
    # Uses inline append rather than _collect_rows because variable names
    # differ from KEY_VARS (validation regression uses different regressors).
    for var in _P02_VAL_VARS:
        if var not in model_p02.params:
            continue
        key_results_rows.append({
            'model':     f'p0_2_validation_t{thresh:.1f}',
            'variable':  var,
            'coeff':     model_p02.params[var],
            't_stat':    model_p02.tvalues[var],
            'p_value':   model_p02.pvalues[var],
            'r_squared': model_p02.rsquared,
            'n_obs':     int(model_p02.nobs),
        })

# Comparison table
print(f"\n  Detector validation (Eq. 5) by threshold | N = {int(p02_records[0]['model'].nobs):,}")
print(f"  {'Threshold':<11} {'High-regime %':>14} {'β₃':>12} {'z-stat':>8} {'p-value':>10} {'Amplif.':>9}")
print(f"  {'-' * 69}")
for r in p02_records:
    sig = '***' if r['p3'] < 0.01 else '**' if r['p3'] < 0.05 else '*' if r['p3'] < 0.10 else ''
    print(f"  {r['thresh']:<11.1f} {r['hi_frac'] * 100:>13.1f}%  "
          f"{r['b3']:>12.6f} {r['z3']:>8.3f} {r['p3']:>10.4f} {sig:<3} "
          f"{r['amp']:>8.3f}x")

# Sanity check at threshold = 0.5
_r05 = next(r for r in p02_records if r['thresh'] == 0.5)
if abs(_r05['b3'] - _P02_ORIG_B3) < 0.0001:
    print(f"\n  Sanity check (threshold=0.5): β₃ = {_r05['b3']:.6f} "
          f"(expected ≈ {_P02_ORIG_B3}) ✓")
else:
    print(f"\n  WARNING: threshold=0.5 β₃ = {_r05['b3']:.6f} "
          f"does not match expected ≈ {_P02_ORIG_B3}.")

# Save: compact comparison table followed by full summaries for all thresholds
_p02_path = os.path.join(RESULTS_DIR, 'p0_2_threshold_sensitivity.txt')
with open(_p02_path, 'w') as _f:
    _f.write("P0-2 THRESHOLD SENSITIVITY — DETECTOR VALIDATION REGRESSION (EQ. 5)\n")
    _f.write("=" * 70 + "\n\n")
    _f.write(f"{'Threshold':<11} {'High-regime %':>14} {'beta_3':>12} "
             f"{'z-stat':>8} {'p-value':>10} {'Amplif.':>9}\n")
    _f.write("-" * 68 + "\n")
    for r in p02_records:
        sig = '***' if r['p3'] < 0.01 else '**' if r['p3'] < 0.05 else '*' if r['p3'] < 0.10 else ''
        _f.write(f"{r['thresh']:<11.1f} {r['hi_frac'] * 100:>13.1f}%  "
                 f"{r['b3']:>12.6f} {r['z3']:>8.3f} {r['p3']:>10.4f} {sig:<3} "
                 f"{r['amp']:>8.3f}x\n")
    for r in p02_records:
        _f.write(f"\n{'=' * 60}\n")
        _f.write(f"Full summary — threshold = {r['thresh']:.1f}\n")
        _f.write(f"{'=' * 60}\n")
        _f.write(str(r['model'].summary()))
        _f.write("\n")
print(f"  Saved: p0_2_threshold_sensitivity.txt")

# =============================================================================
# P0-3 — ADDITIVE COMBINATION ROBUSTNESS                         [PENDING]
# =============================================================================
# Report the additive RegimeScore result for transparency. Construct
# additive RegimeScore as [logistic(z_lambda) + logistic(z_TAR)] / 2,
# re-run primary regression, frame as a transparency comparison.
#
# Required outputs:
#   p0_3_additive_regression.txt
#   p0_key_results.csv  (rows appended)

print("\n" + "=" * 60)
print("P0-3 — ADDITIVE COMBINATION ROBUSTNESS                    [PENDING]")
print("=" * 60)
print("  Not yet implemented — skipped.")

# =============================================================================
# P0-4 — LAMBDA AND TAR WINDOW SENSITIVITY                       [PENDING]
# =============================================================================
# Re-run the primary regression at lambda window lengths of 15, 30, and
# 60 bars (TAR fixed at 5) and at TAR window lengths of 3, 5, and 10
# (lambda fixed at 30). Also re-run the stable-conditions analysis at
# each lambda window length.
#
# Required outputs:
#   p0_4_window_sensitivity.csv
#   p0_key_results.csv  (rows appended)

print("\n" + "=" * 60)
print("P0-4 — LAMBDA AND TAR WINDOW SENSITIVITY                  [PENDING]")
print("=" * 60)
print("  Not yet implemented — skipped.")

# =============================================================================
# P0-5 — EXPANDED ANNOUNCEMENT EXCLUSION SET                     [PENDING]
# =============================================================================
# Expand the announcement exclusion set to six high-impact releases
# (FOMC, CPI, NFP, PPI, advance GDP, retail sales) per Andersen et al.
# (2007). Re-run primary regression with expanded +30-min post-
# announcement exclusion. Report change in N, β₃, and p-value.
#
# Required outputs:
#   p0_5_expanded_exclusion.txt
#   p0_key_results.csv  (rows appended)

print("\n" + "=" * 60)
print("P0-5 — EXPANDED ANNOUNCEMENT EXCLUSION SET                [PENDING]")
print("=" * 60)
print("  Not yet implemented — skipped.")

# =============================================================================
# P0-6 — PRE-ANNOUNCEMENT WINDOW CHARACTERIZATION                [PENDING]
# =============================================================================
# Identify the 30-minute pre-announcement windows for all 6 FOMC events
# in the in-sample period. Compute mean RegimeScore and mean |TFI| in
# these windows vs. full-sample means. Descriptive only — no inference.
#
# Required outputs:
#   p0_6_preannouncement_stats.txt

print("\n" + "=" * 60)
print("P0-6 — PRE-ANNOUNCEMENT WINDOW CHARACTERIZATION           [PENDING]")
print("=" * 60)
print("  Not yet implemented — skipped.")

# =============================================================================
# P0-7 — FORMAL BIAS SIMULATION                                  [PENDING]
# =============================================================================
# Permutation simulation confirming upward bias direction. Generate 1,000
# synthetic datasets under H₀ by permuting forward returns while preserving
# the joint distribution of TFI and RegimeScore. Apply the full primary
# regression pipeline to each permuted dataset. Run on both the full
# in-sample dataset and the stable-conditions subsample (N = 18,355).
#
# Required outputs:
#   p0_7_simulation_full.txt
#   p0_7_simulation_stable.txt
#   p0_7_simulation_histogram.png

print("\n" + "=" * 60)
print("P0-7 — FORMAL BIAS SIMULATION                             [PENDING]")
print("=" * 60)
print("  Not yet implemented — skipped.")

# =============================================================================
# SAVE p0_key_results.csv
# =============================================================================

print("\n" + "=" * 60)
print("SAVING p0_key_results.csv")
print("=" * 60)

if key_results_rows:
    df_key   = pd.DataFrame(key_results_rows)
    csv_path = os.path.join(RESULTS_DIR, 'p0_key_results.csv')
    df_key.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"  Saved: p0_key_results.csv  ({len(key_results_rows)} rows, "
          f"{df_key['model'].nunique()} model(s))")
else:
    print("  No results collected — p0_key_results.csv not written.")

print("\n" + "=" * 60)
print("PHASE 0 PIPELINE COMPLETE")
print("=" * 60)
