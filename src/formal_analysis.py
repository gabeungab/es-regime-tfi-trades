"""
formal_analysis.py — Phase 4 formal statistical analysis pipeline.

Runs the complete regression analysis for the ES futures microstructure
study. All tests use the multiplicative regime detector (Kyle's lambda ×
trade arrival rate), post-only announcement exclusion windows (+30 min
after FOMC, CPI, NFP), and Newey-West HAC standard errors (maxlags=5).

Sections (in execution order):
    1.  Data preparation and signal computation
    2.  Regime score diagnostics (autocorrelation, formulation comparison)
    3.  Regime detector validation (contemporaneous slope test)
    4.  Summary statistics
    5.  Primary regression — Return_{t+1}
    6.  Stable regime conditions test
    7.  Contemporaneous regression — Return_t (lagged RegimeScore)
    8.  Horizon analysis — T+5 and T+15
    9.  Subsample stability
    10. Out-of-sample validation — 2026
    11. OOS diagnostic tests
    12. OOS lambda level comparison
    13. Lagged regime conditioning
    14. Midday subsample
    15. TFI quintile interaction
    16. Regime transition dynamics
    17. Transition delta magnitude diagnostic
    18. Transition threshold robustness
    19. Transaction cost analysis
    20. Save results

Outputs written to results/phase4/:
    primary_regression.txt, contemporaneous_regression.txt,
    oos_primary_regression.txt, lagged_regime_regression.txt,
    midday_regression.txt, quintile_interaction_regression.txt,
    transition_dynamics_regression.txt, horizon_t{5,15}_regression.txt,
    subsample_{full,may_sep,oct_dec}_regression.txt,
    key_coefficients.csv, transaction_cost_analysis.txt
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))

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
OOS_DATA_DIR = os.path.expanduser(
    '/Volumes/X9 Pro/raw-market-data/es-futures/trades/GLBX-20260101-20260309/'
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase4')
os.makedirs(RESULTS_DIR, exist_ok=True)

TZ = 'America/New_York'

# Scheduled macro announcement datetimes (Eastern).
# Post-announcement exclusion windows (+30 min) are applied after each event.
# Pre-announcement windows are retained as potentially informed.
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

OOS_ANNOUNCEMENT_DATES = [
    # FOMC decisions (2:00 PM ET)
    pd.Timestamp('2026-01-28 14:00', tz=TZ),
    pd.Timestamp('2026-03-18 14:00', tz=TZ),
    # CPI releases (8:30 AM ET)
    pd.Timestamp('2026-01-13 08:30', tz=TZ),
    pd.Timestamp('2026-02-11 08:30', tz=TZ),
    pd.Timestamp('2026-03-11 08:30', tz=TZ),
    # NFP releases (8:30 AM ET)
    pd.Timestamp('2026-01-09 08:30', tz=TZ),
    pd.Timestamp('2026-02-11 08:30', tz=TZ),
    pd.Timestamp('2026-03-06 08:30', tz=TZ),
]

# Columns required for the primary regression — used to drop NaN rows once,
# covering warmup, day boundaries, and forward-return edge effects.
REGRESSION_COLS = [
    'fwd_return', 'tfi', 'regime_score', 'tfi_x_regime',
    'regime_score_lag', 'tfi_x_regime_lag', 'lag_return', 'lag_tfi',
]

# Primary regression regressors — used across all main regressions.
PRIMARY_COLS = ['tfi', 'regime_score', 'tfi_x_regime', 'lag_return', 'lag_tfi']

# Primary variables reported in the collated coefficient CSV.
PRIMARY_VARS = ['const', 'tfi', 'regime_score', 'tfi_x_regime', 'lag_return', 'lag_tfi']

# HAC standard error lag truncation applied uniformly across all regressions.
HAC_LAGS = 5

# Bonferroni-corrected significance threshold for the quintile interaction
# section (5 simultaneous tests, family-wise α = 0.05).
BONFERRONI_ALPHA = 0.05 / 5

# ES futures tick size in index points (used for transaction cost analysis).
ES_TICK = 0.25

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _rolling_zscore(series, window):
    """Compute rolling z-score with minimum periods equal to window."""
    mean = series.rolling(window=window, min_periods=window).mean()
    std  = series.rolling(window=window, min_periods=window).std()
    return (series - mean) / std.replace(0, float('nan'))


def _fit_ols(y, X_cols, data):
    """Fit HAC-robust OLS and return the fitted model."""
    X = sm.add_constant(data[X_cols])
    return sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})


def _print_coeff_table(model, variables):
    """Print a compact coefficient table for the specified variables."""
    print(f"  {'Variable':<20} {'Coeff':>12} {'z-stat':>8} {'p-value':>10}")
    print(f"  {'-' * 54}")
    for var in variables:
        if var not in model.params:
            continue
        c   = model.params[var]
        t   = model.tvalues[var]
        p   = model.pvalues[var]
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
        print(f"  {var:<20} {c:>12.6f} {t:>8.3f} {p:>10.4f} {sig}")
    print(f"  R²: {model.rsquared:.6f}")


def _save_model(model, filename):
    """Write a statsmodels summary to a text file in RESULTS_DIR."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, 'w') as f:
        f.write(str(model.summary()))
    print(f"  Saved: {filename}")


def _collect_rows(fitted_model, model_label, rows):
    """
    Append coefficient rows for PRIMARY_VARS from fitted_model to rows.
    Used to build the collated key_coefficients.csv output.
    """
    for var in PRIMARY_VARS:
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
    Assemble the regression DataFrame from signal outputs, null out
    first bars of each day (overnight return contamination), and
    construct all lagged and interaction terms.

    Accepts either a Series or a single-column DataFrame for each signal.

    Returns
    -------
    pd.DataFrame
        Columns: tfi, log_return, regime_score, fwd_return, lag_return,
        lag_tfi, tfi_x_regime, regime_score_lag, tfi_x_regime_lag.
    """
    tfi_s = tfi_input['tfi'] if isinstance(tfi_input, pd.DataFrame) else tfi_input
    ret_s = (returns_input['log_return']
             if isinstance(returns_input, pd.DataFrame) else returns_input)

    df = pd.DataFrame({
        'tfi':          tfi_s,
        'log_return':   ret_s,
        'regime_score': regime_score_input,
    })

    # Null the first bar of each trading day — the overnight gap means
    # log_return at bar 0 is not a valid RTH intraday return.
    dates = pd.Series(df.index.date, index=df.index)
    df.loc[dates != dates.shift(1), 'log_return'] = np.nan

    df['fwd_return']       = df['log_return'].shift(-1)
    df['lag_return']       = df['log_return']
    df['lag_tfi']          = df['tfi'].shift(1)
    df['tfi_x_regime']     = df['tfi'] * df['regime_score']
    df['regime_score_lag'] = df['regime_score'].shift(1)
    df['tfi_x_regime_lag'] = df['tfi'] * df['regime_score_lag']

    return df


def _compute_signed_flow(df_clean):
    """
    Compute per-bar signed order flow (buy volume minus sell volume)
    at 1-minute resolution using aggressor-side trade classification.
    Used as the input for the lambda window stability metric.

    Returns
    -------
    pd.Series
        Signed order flow at each 1-minute bar, indexed by ts_event_et.
    """
    indexed = df_clean.set_index('ts_event_et')
    return (
        indexed.groupby(pd.Grouper(freq='1min'))
        .apply(lambda x: float(x.loc[x['side'] == 'B', 'size'].sum())
                       - float(x.loc[x['side'] == 'A', 'size'].sum()))
        .rename('signed_flow')
    )


# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

print("=" * 60)
print("PHASE 4 — FORMAL STATISTICAL ANALYSIS")
print("=" * 60)

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

# Strip timezone if bar index is tz-naive (environment-dependent).
ann_dates = ANNOUNCEMENT_DATES
if bars.index.tzinfo is None:
    ann_dates = [dt.tz_localize(None) for dt in ann_dates]

exclusion_mask = compute_exclusion_mask(bars, ann_dates)
regime_score   = compute_regime_score(lambda_series, arrival_series, exclusion_mask)
tfi            = compute_tfi(df_clean)
returns        = compute_returns(df_clean)

# Signed flow used for the lambda window stability metric —
# rolling std of signed flow over the same 30-bar window used by lambda.
signed_flow_series = _compute_signed_flow(df_clean)

print("\n[3] Building regression DataFrame...")
reg   = _build_reg_df(tfi, returns, regime_score)
n_raw = len(reg)
reg   = reg.dropna(subset=REGRESSION_COLS)
print(f"    Bars before filters:             {n_raw:,}")
print(f"    Dropped (NaN/warmup/boundaries): {n_raw - len(reg):,}")
print(f"    Final regression N:              {len(reg):,}")

# =============================================================================
# 2. REGIME SCORE AUTOCORRELATION DIAGNOSTIC
# =============================================================================
# Informs interpretation of the contemporaneous regression.
# High lag-1 autocorrelation means lagging RegimeScore by one bar removes
# little circularity — residual confounding dominates the contemporaneous
# p-value regardless of the regime-lagging specification.

print("\n" + "=" * 60)
print("REGIME SCORE AUTOCORRELATION DIAGNOSTIC")
print("=" * 60)

ac_lag1  = reg['regime_score'].autocorr(lag=1)
ac_lag5  = reg['regime_score'].autocorr(lag=5)
ac_lag30 = reg['regime_score'].autocorr(lag=30)

print(f"\n  RegimeScore autocorrelation:")
print(f"    Lag 1:  {ac_lag1:.4f}")
print(f"    Lag 5:  {ac_lag5:.4f}")
print(f"    Lag 30: {ac_lag30:.4f}")
print(f"\n  Interpretation:")
print(f"    > 0.85 → lagging by 1 bar removes little circularity;")
print(f"             residual confounding dominates contemporaneous p-value")
print(f"    < 0.70 → meaningful circularity removed;")
print(f"             contemporaneous p-value may reflect genuine signal")

# =============================================================================
# DETECTOR FORMULATION COMPARISON — MULTIPLICATIVE VS ADDITIVE
# =============================================================================
# Compares the multiplicative RegimeScore (current) against the prior additive
# formulation to quantify how much the design change affects the signal.
# High correlation indicates findings are robust to the formulation choice.
# Low correlation indicates materially different regime classifications.

print("\n" + "=" * 60)
print("DETECTOR FORMULATION COMPARISON — MULTIPLICATIVE VS ADDITIVE")
print("=" * 60)

# Recompute additive score on same lambda and arrival series
z_lambda_add  = _rolling_zscore(lambda_series,  window=30)
z_arrival_add = _rolling_zscore(arrival_series, window=5)
composite_add = z_lambda_add + z_arrival_add
regime_score_additive = 1 / (1 + np.exp(-composite_add))

# Apply same exclusion mask
excl_aligned          = exclusion_mask.reindex(regime_score_additive.index, fill_value=False)
regime_score_additive = regime_score_additive.where(~excl_aligned, other=0.0)

# Align both to regression index for fair comparison
rs_mult = reg['regime_score']
rs_add  = regime_score_additive.reindex(reg.index)
valid   = rs_mult.notna() & rs_add.notna()
corr    = rs_mult[valid].corr(rs_add[valid])

print(f"\n  Pearson correlation (multiplicative vs additive): {corr:.4f}")
print(f"\n  {'Metric':<30} {'Multiplicative':>16} {'Additive':>16}")
print(f"  {'-' * 64}")
for metric, fn in [('Mean',        lambda s: s.mean()),
                   ('Std',         lambda s: s.std()),
                   ('Median',      lambda s: s.median()),
                   ('75th pctile', lambda s: s.quantile(0.75))]:
    print(f"  {metric:<30} {fn(rs_mult[valid]):>16.4f} {fn(rs_add[valid]):>16.4f}")

hi_mult   = (rs_mult > 0.5).mean()
hi_add    = (rs_add  > 0.5).mean()
agreement = ((rs_mult > 0.5) == (rs_add > 0.5))[valid].mean()

print(f"\n  High-regime fraction (>0.5):")
print(f"    Multiplicative: {hi_mult:.4f} ({hi_mult * 100:.1f}%)")
print(f"    Additive:       {hi_add:.4f}  ({hi_add  * 100:.1f}%)")
print(f"\n  Bar-level regime classification agreement: "
      f"{agreement:.4f} ({agreement * 100:.1f}%)")
print(f"\n  Interpretation:")
print(f"    Correlation > 0.95 → formulations are equivalent in practice;")
print(f"                         findings are robust to design choice.")
print(f"    Correlation < 0.90 → material difference; rerun all tests")
print(f"                         and compare findings explicitly.")

# =============================================================================
# 3. REGIME DETECTOR VALIDATION
# =============================================================================
# Formal validation test: contemporaneous TFI-return slope should be
# statistically significantly larger in high-regime bars than low-regime bars.
# This is the direct empirical implication of Kyle (1985) — elevated adverse
# selection produces stronger price impact per unit of order flow.
#
# Note: a pre-announcement RegimeScore elevation test was also considered
# but was deemed infeasible with RTH-only data (only 6 observable FOMC
# pre-announcement windows). Results from that test are not reported.

print("\n" + "=" * 60)
print("REGIME DETECTOR VALIDATION")
print("=" * 60)

print("\n--- Validation Test: Contemporaneous TFI-Return Slope by Regime ---")
print("  Tests whether the within-bar TFI-return relationship is")
print("  significantly stronger in high-regime bars than low-regime bars.")
print("  A significant interaction confirms the detector correctly")
print("  identifies periods of elevated within-bar price impact.")

# RegimeScore_t is used contemporaneously (not lagged) because the validation
# specifically asks whether the current bar's regime state is associated with
# stronger within-bar price impact. Using lagged RegimeScore would test a
# weaker, less direct question. TFI-Return confounding is acknowledged —
# this is a directional sanity check, not a causal test.

reg_val = reg.copy()
reg_val['high_regime_dummy'] = (reg_val['regime_score'] > 0.5).astype(float)
reg_val['tfi_x_high_regime'] = reg_val['tfi'] * reg_val['high_regime_dummy']

val_cols   = ['tfi', 'high_regime_dummy', 'tfi_x_high_regime', 'lag_tfi']
y_val      = reg_val['lag_return']   # Return_t (contemporaneous)
X_val      = sm.add_constant(reg_val[val_cols])
model_val1 = sm.OLS(y_val, X_val).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

b1_val     = model_val1.params['tfi']
b3_val     = model_val1.params['tfi_x_high_regime']
p3_val     = model_val1.pvalues['tfi_x_high_regime']
z3_val     = model_val1.tvalues['tfi_x_high_regime']
slope_low  = b1_val
slope_high = b1_val + b3_val

print(f"\n  Contemporaneous validation regression | N={int(model_val1.nobs):,}")
print(f"  {'Variable':<25} {'Coeff':>12} {'z-stat':>8} {'p-value':>10}")
print(f"  {'-' * 59}")
for var in ['const', 'tfi', 'high_regime_dummy', 'tfi_x_high_regime', 'lag_tfi']:
    c   = model_val1.params[var]
    t   = model_val1.tvalues[var]
    p   = model_val1.pvalues[var]
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
    print(f"  {var:<25} {c:>12.6f} {t:>8.3f} {p:>10.4f} {sig}")

print(f"\n  TFI-return slope breakdown:")
print(f"    Low-regime  (RegimeScore ≤ 0.5): {slope_low:.6f}  (= β₁)")
print(f"    High-regime (RegimeScore > 0.5): {slope_high:.6f}  (= β₁ + β₃)")
if slope_low != 0:
    print(f"    Amplification ratio: {slope_high / slope_low:.3f}x")

print(f"\n  Interaction term (β₃ = tfi_x_high_regime):")
print(f"    Coeff:   {b3_val:.6f}")
print(f"    z-stat:  {z3_val:.3f}")
print(f"    p-value: {p3_val:.4f}")

if p3_val < 0.05:
    print(f"  → SIGNIFICANT: High-regime slope is statistically larger.")
    print(f"    Detector correctly identifies elevated within-bar price impact.")
else:
    print(f"  → NOT SIGNIFICANT: Cannot confirm regime-conditioned slope difference.")
    print(f"    Note: result is directionally consistent if high-regime slope > low.")

print(f"\n  NOTE: This test uses contemporaneous RegimeScore_t — TFI-Return")
print(f"  confounding within bar t is present and acknowledged. This is a")
print(f"  directional sanity check, not a causal test.")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n[4] Summary statistics — regression sample:")
print("-" * 60)
print(reg[REGRESSION_COLS].describe().round(6).to_string())

high_n  = (reg['regime_score'] > 0.5).sum()
low_n   = (reg['regime_score'] <= 0.5).sum()
n_final = len(reg)

print(f"\n  Regime distribution:")
print(f"    High-regime (RegimeScore > 0.5): {high_n:,} ({high_n / n_final * 100:.1f}%)")
print(f"    Low-regime  (RegimeScore ≤ 0.5): {low_n:,}  ({low_n  / n_final * 100:.1f}%)")
print(f"\n  Date range:")
print(f"    First bar:    {reg.index.min()}")
print(f"    Last bar:     {reg.index.max()}")
print(f"    Trading days: {reg.index.normalize().nunique()}")

# =============================================================================
# 4. PRIMARY REGRESSION — Return_{t+1}
# =============================================================================
# Primary test: does the regime score amplify TFI's forward predictive power?
# RegimeScore_t is contemporaneous with TFI_t but fully predetermined relative
# to Return_{t+1}, making this regression non-confounded in the forward-return
# sense. Circularity at the interaction level remains — see paper Section 4.5.

print("\n" + "=" * 60)
print("PRIMARY REGRESSION — Return_{t+1}")
print("=" * 60)

y     = reg['fwd_return']
model = _fit_ols(y, PRIMARY_COLS, reg)
print(model.summary())

# =============================================================================
# 5. STABLE REGIME CONDITIONS TEST
# =============================================================================
# Tests whether the regime detector's signal is stronger when lambda estimation
# conditions are most stable. Stability is proxied by the rolling standard
# deviation of signed order flow in the 30-bar lambda estimation window —
# the same window used by the regime detector. Low std means the window
# contains steady, representative flow rather than a single large directional
# spike that would dominate the rolling OLS estimate.
# Bottom tercile = most stable estimation conditions.

print("\n" + "=" * 60)
print("STABLE REGIME CONDITIONS TEST")
print("=" * 60)

signed_flow_aligned      = signed_flow_series.reindex(reg.index)
reg['lambda_window_std'] = (
    signed_flow_aligned
    .rolling(30, min_periods=15)
    .std()
    .reindex(reg.index)
)

stability_p33 = reg['lambda_window_std'].quantile(0.33)
stability_p67 = reg['lambda_window_std'].quantile(0.67)

reg_stable   = reg[reg['lambda_window_std'] <= stability_p33].dropna(subset=REGRESSION_COLS)
reg_unstable = reg[reg['lambda_window_std'] >  stability_p67].dropna(subset=REGRESSION_COLS)

print(f"\n  Stability threshold (33rd pctile): {stability_p33:.2f} contracts signed-flow std")
print(f"  Stable-condition bars:   {len(reg_stable):,} ({len(reg_stable) / n_final * 100:.1f}% of sample)")
print(f"  High-regime in stable:   "
      f"{(reg_stable['regime_score'] > 0.5).sum():,} "
      f"({(reg_stable['regime_score'] > 0.5).mean() * 100:.1f}%)")

print(f"\n  Hour distribution of stable bars:")
stable_hour_dist = reg_stable.index.hour.value_counts().sort_index()
for h, n in stable_hour_dist.items():
    pct = n / len(reg_stable) * 100
    bar = '█' * int(pct / 1.5)
    print(f"    {h:02d}:xx  {n:5,} ({pct:4.1f}%) {bar}")

model_stable   = _fit_ols(reg_stable['fwd_return'],   PRIMARY_COLS, reg_stable)
model_unstable = _fit_ols(reg_unstable['fwd_return'], PRIMARY_COLS, reg_unstable)

print(f"\n  Stable-condition regression | N={int(model_stable.nobs):,}")
print(f"    β₃ (tfi_x_regime): {model_stable.params['tfi_x_regime']:.6f}")
print(f"    z-stat:            {model_stable.tvalues['tfi_x_regime']:.3f}")
print(f"    p-value:           {model_stable.pvalues['tfi_x_regime']:.4f}")
print(f"    β₁ (tfi):          {model_stable.params['tfi']:.6f}")
print(f"    R²:                {model_stable.rsquared:.6f}")

print(f"\n  Unstable-condition regression | N={int(model_unstable.nobs):,}")
print(f"    β₃ (tfi_x_regime): {model_unstable.params['tfi_x_regime']:.6f}")
print(f"    p-value:           {model_unstable.pvalues['tfi_x_regime']:.4f}")

print(f"\n  Full-sample reference:    β₃={model.params['tfi_x_regime']:.6f}  "
      f"p={model.pvalues['tfi_x_regime']:.4f}")
direction = ("CONSISTENT" if model_stable.pvalues['tfi_x_regime']
             < model.pvalues['tfi_x_regime'] else "NOT consistent")
print(f"  Gradient (stable < full < unstable p-value): {direction} with detector quality hypothesis")

# =============================================================================
# 6. CONTEMPORANEOUS REGRESSION — Return_t (lagged RegimeScore)
# =============================================================================
# Secondary specification: tests for same-bar regime amplification.
# RegimeScore is lagged by one bar to remove regime-level simultaneity.
# TFI-Return confounding within bar t remains — irreducible with trades-only
# data. This is a specification sensitivity test, not a predictive test.

print("\n" + "=" * 60)
print("CONTEMPORANEOUS REGRESSION — Return_t (lagged RegimeScore)")
print("=" * 60)

y_contemp     = reg['lag_return']
model_contemp = _fit_ols(y_contemp,
                         ['tfi', 'regime_score_lag', 'tfi_x_regime_lag', 'lag_tfi'],
                         reg)
print(model_contemp.summary())

# =============================================================================
# 7. HORIZON ANALYSIS — T+5 and T+15 Cumulative Returns
# =============================================================================
# Tests whether any regime-conditioned predictability persists beyond one bar.
# Cumulative log returns are computed within each trading day only —
# cross-day windows are nulled to avoid overnight return contamination.

print("\n" + "=" * 60)
print("HORIZON ANALYSIS — T+5 and T+15 Cumulative Returns")
print("=" * 60)

horizon_models   = {}
reg_dates_series = pd.Series(reg.index.date, index=reg.index)

for h in [5, 15]:
    col_h      = f'fwd_return_{h}'
    reg[col_h] = reg['log_return'].rolling(h).sum().shift(-h)

    # Null any window that crosses a day boundary
    for i in range(1, h + 1):
        crosses = reg_dates_series != reg_dates_series.shift(-i)
        reg.loc[crosses, col_h] = np.nan

    horizon_cols      = [col_h, 'tfi', 'regime_score', 'tfi_x_regime', 'lag_return', 'lag_tfi']
    reg_h             = reg.dropna(subset=horizon_cols)
    model_h           = _fit_ols(reg_h[col_h], PRIMARY_COLS, reg_h)
    horizon_models[h] = model_h

    print(f"\n  Horizon T+{h} | N={len(reg_h):,}")
    _print_coeff_table(model_h, ['const', 'tfi', 'regime_score', 'tfi_x_regime',
                                  'lag_return', 'lag_tfi'])

# =============================================================================
# 8. SUBSAMPLE STABILITY
# =============================================================================
# Splits the in-sample period into two activity regimes to test whether the
# primary null result is stable. May–Sep = lower-volatility period;
# Oct–Dec = elevated volatility with more macro events.

print("\n" + "=" * 60)
print("SUBSAMPLE STABILITY")
print("=" * 60)

subsample_models = {}
subsamples = {
    'full':    (None,                              None),
    'may_sep': (pd.Timestamp('2025-05-01').date(), pd.Timestamp('2025-09-30').date()),
    'oct_dec': (pd.Timestamp('2025-10-01').date(), pd.Timestamp('2025-12-31').date()),
}
period_labels = {
    'full':    'Full sample (May–Dec 2025)',
    'may_sep': 'May–Sep 2025',
    'oct_dec': 'Oct–Dec 2025',
}

for key, (start, end) in subsamples.items():
    if start is None:
        reg_sub = reg.copy()
    else:
        idx_dates = pd.Series(reg.index.date, index=reg.index)
        reg_sub   = reg[(idx_dates >= start) & (idx_dates <= end)]

    reg_sub = reg_sub.dropna(subset=REGRESSION_COLS)
    if len(reg_sub) < 100:
        print(f"\n  {key}: insufficient observations ({len(reg_sub)}), skipping")
        continue

    model_sub             = _fit_ols(reg_sub['fwd_return'], PRIMARY_COLS, reg_sub)
    subsample_models[key] = model_sub

    print(f"\n  {period_labels[key]} | N={len(reg_sub):,}")
    _print_coeff_table(model_sub, ['const', 'tfi', 'regime_score', 'tfi_x_regime',
                                    'lag_return', 'lag_tfi'])

# =============================================================================
# 9. OUT-OF-SAMPLE VALIDATION — 2026
# =============================================================================
# Applies the in-sample specification to held-out 2026 data without refitting
# any parameters. The same regime detector and announcement exclusion logic
# are applied using 2026-specific announcement dates.

print("\n" + "=" * 60)
print("OUT-OF-SAMPLE VALIDATION — 2026 Jan–Mar")
print("=" * 60)

print("\n  [OOS-1] Loading OOS data...")
df_oos       = load_all_days(OOS_DATA_DIR)
df_oos_clean = remove_outliers(df_oos)
print(f"    {len(df_oos_clean):,} clean RTH trades across "
      f"{df_oos_clean['ts_event_et'].dt.date.nunique()} trading days")

print("  [OOS-2] Computing OOS signals...")
lambda_oos  = compute_lambda(df_oos_clean)
arrival_oos = compute_arrival_rate(df_oos_clean)

df_oos_indexed = df_oos_clean.set_index('ts_event_et')
bars_oos       = df_oos_indexed['price'].resample('1min').count()

ann_dates_oos = OOS_ANNOUNCEMENT_DATES
if bars_oos.index.tzinfo is None:
    ann_dates_oos = [dt.tz_localize(None) for dt in ann_dates_oos]

exclusion_mask_oos = compute_exclusion_mask(bars_oos, ann_dates_oos)
regime_score_oos   = compute_regime_score(lambda_oos, arrival_oos, exclusion_mask_oos)
tfi_oos            = compute_tfi(df_oos_clean)
returns_oos        = compute_returns(df_oos_clean)

reg_oos = _build_reg_df(tfi_oos, returns_oos, regime_score_oos)
reg_oos = reg_oos.dropna(subset=REGRESSION_COLS)

print(f"\n  OOS regression N: {len(reg_oos):,}")
print(f"  OOS date range:   {reg_oos.index.min()} → {reg_oos.index.max()}")

# ── Return scale verification ─────────────────────────────────────────────────
# Confirms OOS returns are on the same scale as in-sample before interpreting
# coefficient magnitudes. Price range check rules out unit or contract issues.

print("\n  --- Return Scale Verification ---")
print(f"  {'':25} {'In-sample':>12} {'OOS':>12}")
print(f"  {'-' * 51}")
print(f"  {'Mean |return|':<25} "
      f"{reg['lag_return'].abs().mean():>12.8f} "
      f"{reg_oos['lag_return'].abs().mean():>12.8f}")
print(f"  {'Return std':<25} "
      f"{reg['lag_return'].std():>12.8f} "
      f"{reg_oos['lag_return'].std():>12.8f}")
print(f"  {'OOS/IS mean |return| ratio':<25} "
      f"{reg_oos['lag_return'].abs().mean() / reg['lag_return'].abs().mean():>12.4f}")
print(f"\n  Price range — in-sample: {df_indexed['price'].min():.2f} – "
      f"{df_indexed['price'].max():.2f}")
print(f"  Price range — OOS:       {df_oos_indexed['price'].min():.2f} – "
      f"{df_oos_indexed['price'].max():.2f}")

# ── OOS primary regression ────────────────────────────────────────────────────
model_oos     = _fit_ols(reg_oos['fwd_return'], PRIMARY_COLS, reg_oos)
oos_b3_pvalue = model_oos.pvalues['tfi_x_regime']

print(f"\n  OOS Primary Regression | N={int(model_oos.nobs):,}")
print(f"    β₃ (tfi_x_regime): {model_oos.params['tfi_x_regime']:.6f}")
print(f"    z-stat:            {model_oos.tvalues['tfi_x_regime']:.3f}")
print(f"    p-value:           {oos_b3_pvalue:.4f}")

# =============================================================================
# 10. OOS DIAGNOSTIC TESTS
# =============================================================================
# Seven diagnostics test why the OOS period produces a significant result
# when the in-sample result is null. Each targets a specific alternative
# explanation: HAC adequacy, regime distribution shift, volatility inflation,
# monthly concentration, rolling episode detection, TFI extremity, and
# permutation-based spuriousness.

print("\n" + "=" * 60)
print(f"OOS DIAGNOSTIC TESTS — Investigating p={oos_b3_pvalue:.4f} Significance")
print("=" * 60)

# ── Diagnostic 1: Residual autocorrelation ────────────────────────────────────
print("\n--- Diagnostic 1: OOS Residual Autocorrelation ---")
print("  Tests whether HAC maxlags=5 adequately corrects OOS standard errors.")
print("  Autocorrelation at lags 6–10 would indicate underestimated z-stats.")

for lag in range(1, 11):
    ac   = model_oos.resid.autocorr(lag=lag)
    flag = ' ← beyond maxlags' if lag > 5 else ''
    print(f"    Lag {lag:2d}: {ac:+.4f}{flag}")

# ── Diagnostic 2: RegimeScore distribution ────────────────────────────────────
print("\n--- Diagnostic 2: RegimeScore Distribution Comparison ---")
print("  Tests whether OOS regime distribution is drawn from the same")
print("  population as in-sample. A structural shift would indicate the")
print("  regime detector is operating in a materially different environment.")

is_rs  = reg['regime_score'].describe()
oos_rs = reg_oos['regime_score'].describe()

print(f"\n  {'Metric':<20} {'In-sample':>12} {'OOS':>12} {'Difference':>12}")
print(f"  {'-' * 58}")
for metric in ['mean', 'std', '25%', '50%', '75%']:
    diff = oos_rs[metric] - is_rs[metric]
    print(f"  {metric:<20} {is_rs[metric]:>12.4f} {oos_rs[metric]:>12.4f} {diff:>+12.4f}")

is_hi_pct  = (reg['regime_score']     > 0.5).mean()
oos_hi_pct = (reg_oos['regime_score'] > 0.5).mean()
print(f"\n  High-regime fraction (>0.5):")
print(f"    In-sample: {is_hi_pct:.4f} ({is_hi_pct * 100:.1f}%)")
print(f"    OOS:       {oos_hi_pct:.4f} ({oos_hi_pct * 100:.1f}%)")
print(f"    OOS/IS ratio: {oos_hi_pct / is_hi_pct:.3f}x  (>1.10x would be a concern)")

# ── Diagnostic 3: Realized volatility ────────────────────────────────────────
print("\n--- Diagnostic 3: Realized Volatility Comparison ---")
print("  Tests whether elevated OOS volatility mechanically inflates lambda")
print("  and RegimeScore, producing a spurious interaction coefficient.")

is_vol  = reg['lag_return'].std()
oos_vol = reg_oos['lag_return'].std()

print(f"\n  1-minute return std:")
print(f"    In-sample: {is_vol:.6f} ({is_vol * 1e4:.3f} bps per bar)")
print(f"    OOS:       {oos_vol:.6f} ({oos_vol * 1e4:.3f} bps per bar)")
print(f"    OOS/IS ratio: {oos_vol / is_vol:.3f}x")

print(f"\n  Mean RegimeScore (lambda-level proxy):")
print(f"    In-sample: {reg['regime_score'].mean():.4f}")
print(f"    OOS:       {reg_oos['regime_score'].mean():.4f}")
print(f"    OOS/IS ratio: {reg_oos['regime_score'].mean() / reg['regime_score'].mean():.3f}x")

# ── Diagnostic 4: OOS result by month ────────────────────────────────────────
print("\n--- Diagnostic 4: OOS Result by Month ---")
print("  Tests whether aggregate OOS significance is driven by one month.")

oos_months = {
    'January 2026':  (pd.Timestamp('2026-01-01').date(), pd.Timestamp('2026-01-31').date()),
    'February 2026': (pd.Timestamp('2026-02-01').date(), pd.Timestamp('2026-02-28').date()),
}
oos_date_col = pd.Series(reg_oos.index.date, index=reg_oos.index)

for month_label, (start, end) in oos_months.items():
    reg_month = reg_oos[
        (oos_date_col >= start) & (oos_date_col <= end)
    ].dropna(subset=REGRESSION_COLS)
    if len(reg_month) < 200:
        print(f"\n  {month_label}: N={len(reg_month)} — insufficient for regression, skipping")
        continue
    m = _fit_ols(reg_month['fwd_return'], PRIMARY_COLS, reg_month)
    print(f"\n  {month_label} | N={int(m.nobs):,}")
    print(f"    β₃ (tfi_x_regime): {m.params['tfi_x_regime']:.6f}")
    print(f"    z-stat:            {m.tvalues['tfi_x_regime']:.3f}")
    print(f"    p-value:           {m.pvalues['tfi_x_regime']:.4f}")

print(f"\n  Note: March 2026 has only ~3 trading days in OOS window —")
print(f"  insufficient for a separate monthly regression.")

# ── Diagnostic 5: Rolling 10-day windows ─────────────────────────────────────
print("\n--- Diagnostic 5: Rolling 10-Day β₃ Across OOS Period ---")
print("  Tests whether OOS significance is episodic (concentrated in a")
print("  specific window) or structural (stable across the full period).")

oos_dates_series = pd.Series(reg_oos.index.date, index=reg_oos.index)
oos_unique_dates = sorted(oos_dates_series.unique())
WINDOW_DAYS      = 10

print(f"\n  Rolling {WINDOW_DAYS}-trading-day window β₃ estimates:")
print(f"  {'Window end':<15} {'N':>6} {'β₃':>12} {'p-value':>10}")
print(f"  {'-' * 47}")

rolling_results = []
for i in range(WINDOW_DAYS - 1, len(oos_unique_dates)):
    w_start = oos_unique_dates[i - WINDOW_DAYS + 1]
    w_end   = oos_unique_dates[i]
    mask    = (oos_dates_series >= w_start) & (oos_dates_series <= w_end)
    reg_win = reg_oos[mask].dropna(subset=REGRESSION_COLS)
    if len(reg_win) < 150:
        continue
    try:
        m_w = _fit_ols(reg_win['fwd_return'], PRIMARY_COLS, reg_win)
        b3  = m_w.params['tfi_x_regime']
        pv  = m_w.pvalues['tfi_x_regime']
        rolling_results.append((w_end, len(reg_win), b3, pv))
        sig = '***' if pv < 0.01 else '**' if pv < 0.05 else ''
        print(f"  {str(w_end):<15} {len(reg_win):>6,} {b3:>12.6f} {pv:>10.4f} {sig}")
    except Exception:
        continue

if rolling_results:
    b3_vals = [r[2] for r in rolling_results]
    pv_vals = [r[3] for r in rolling_results]
    n_sig   = sum(1 for p in pv_vals if p < 0.05)
    print(f"\n  Summary: {len(rolling_results)} windows tested, "
          f"{n_sig} significant (p<0.05)")
    print(f"  β₃ range: [{min(b3_vals):.6f}, {max(b3_vals):.6f}]")

# ── Diagnostic 6: TFI distribution ───────────────────────────────────────────
print("\n--- Diagnostic 6: OOS TFI Distribution Comparison ---")
print("  Tests whether OOS had more extreme TFI values, which would")
print("  mechanically inflate β₃ via the regime-TFI circularity.")

is_tfi  = reg['tfi'].describe()
oos_tfi = reg_oos['tfi'].describe()

print(f"\n  {'Metric':<20} {'In-sample':>12} {'OOS':>12} {'Difference':>12}")
print(f"  {'-' * 58}")
for metric in ['mean', 'std', '25%', '50%', '75%', 'max']:
    diff = oos_tfi[metric] - is_tfi[metric]
    print(f"  {metric:<20} {is_tfi[metric]:>12.4f} {oos_tfi[metric]:>12.4f} {diff:>+12.4f}")

is_extreme  = (reg['tfi'].abs()     > 0.3).mean()
oos_extreme = (reg_oos['tfi'].abs() > 0.3).mean()
print(f"\n  Fraction |TFI| > 0.3 (extreme imbalance):")
print(f"    In-sample: {is_extreme:.4f} ({is_extreme * 100:.1f}%)")
print(f"    OOS:       {oos_extreme:.4f} ({oos_extreme * 100:.1f}%)")
if oos_extreme > is_extreme * 1.1:
    print(f"  → OOS has meaningfully more extreme TFI: consistent with")
    print(f"    mechanical inflation via circularity.")
else:
    print(f"  → OOS TFI distribution is similar to in-sample.")

# ── Diagnostic 7: Permutation test ───────────────────────────────────────────
print("\n--- Diagnostic 7: Permutation Test (N=1,000) ---")
print("  Tests whether OOS β₃ is in the tail of the null distribution.")
print("  Returns are randomly permuted, destroying return predictability")
print("  while preserving the regressor structure.")

np.random.seed(42)
N_PERMS   = 1_000
perm_b3   = []
y_oos_arr = reg_oos['fwd_return'].values.copy()
X_oos_arr = sm.add_constant(reg_oos[PRIMARY_COLS]).values

for _ in range(N_PERMS):
    y_perm = np.random.permutation(y_oos_arr)
    try:
        b = np.linalg.lstsq(X_oos_arr, y_perm, rcond=None)[0]
        perm_b3.append(b[3])   # index 3 = tfi_x_regime (after intercept)
    except Exception:
        continue

perm_b3   = np.array(perm_b3)
actual_b3 = model_oos.params['tfi_x_regime']
perm_pval = (perm_b3 >= actual_b3).mean()

print(f"\n  Actual OOS β₃:        {actual_b3:.6f}")
print(f"  Permutation null β₃:  mean={perm_b3.mean():.6f},  std={perm_b3.std():.6f}")
print(f"  Permutation p-value:  {perm_pval:.4f}")
print(f"  Parametric HAC p-value: {model_oos.pvalues['tfi_x_regime']:.4f}")

if perm_pval < 0.05:
    print(f"  → Actual β₃ is in the top {perm_pval * 100:.1f}% of the null distribution.")
    print(f"    OOS result is NOT purely spurious.")
else:
    print(f"  → Actual β₃ is NOT in the tail of the null distribution.")
    print(f"    Result may reflect serial dependence rather than a true effect.")
print(f"\n  Note: permutation destroys time-series dependence. A non-significant")
print(f"  permutation p-value may reflect serial correlation, not a false positive.")

# ── Late vs Early OOS Decomposition ──────────────────────────────────────────
# Splits the OOS period at February 23, 2026 — the date from which rolling
# windows begin showing consecutive significance in Diagnostic 5.

print("\n--- OOS Late vs Early Period Decomposition ---")
print("  Tests whether aggregate OOS significance is driven by the final")
print("  two weeks (Feb 23–Mar 6), which showed consecutive significant")
print("  windows in Diagnostic 5.")

late_feb_start   = pd.Timestamp('2026-02-23').date()
oos_dates_series = pd.Series(reg_oos.index.date, index=reg_oos.index)
reg_oos_early    = reg_oos[oos_dates_series <  late_feb_start].dropna(subset=REGRESSION_COLS)
reg_oos_late     = reg_oos[oos_dates_series >= late_feb_start].dropna(subset=REGRESSION_COLS)

for label, subset in [('Early OOS (Jan–Feb 22)',  reg_oos_early),
                       ('Late OOS  (Feb 23–Mar 6)', reg_oos_late)]:
    print(f"\n  {label} | N={len(subset):,}")
    print(f"    Mean RegimeScore: {subset['regime_score'].mean():.4f}  "
          f"High-regime: {(subset['regime_score'] > 0.5).mean() * 100:.1f}%")
    print(f"    Return std (bps): {subset['lag_return'].std() * 1e4:.3f}  "
          f"Mean |TFI|: {subset['tfi'].abs().mean():.4f}")
    m = _fit_ols(subset['fwd_return'], PRIMARY_COLS, subset)
    print(f"    β₃: {m.params['tfi_x_regime']:.6f}  "
          f"z={m.tvalues['tfi_x_regime']:.3f}  "
          f"p={m.pvalues['tfi_x_regime']:.4f}  "
          f"R²={m.rsquared:.6f}")

# ── OOS Lambda Window Stability ───────────────────────────────────────────────
# Tests whether the late OOS significance is linked to more stable lambda
# estimation conditions — the same hypothesis motivating the in-sample
# stable regime conditions test.

print("\n--- OOS Lambda Window Stability: Late vs Early ---")

signed_flow_oos       = _compute_signed_flow(df_oos_clean)
signed_flow_oos_align = signed_flow_oos.reindex(reg_oos.index)
reg_oos['lambda_window_std'] = (
    signed_flow_oos_align
    .rolling(30, min_periods=15)
    .std()
    .reindex(reg_oos.index)
)

late_mask     = oos_dates_series >= late_feb_start
early_lws_std = reg_oos.loc[~late_mask, 'lambda_window_std']
late_lws_std  = reg_oos.loc[late_mask,  'lambda_window_std']

print(f"\n  Early OOS lambda_window_std:")
print(early_lws_std.describe().to_string())
print(f"\n  Late OOS lambda_window_std:")
print(late_lws_std.describe().to_string())
print(f"\n  Early mean: {early_lws_std.mean():.2f}  "
      f"Late mean: {late_lws_std.mean():.2f}  "
      f"Ratio (late/early): {late_lws_std.mean() / early_lws_std.mean():.3f}x")

print(f"\n--- Hour Distribution: Late OOS Period ---")
late_hours = reg_oos.loc[late_mask].index.hour.value_counts().sort_index()
n_late     = late_mask.sum()
for h, n in late_hours.items():
    pct = n / n_late * 100
    bar = '█' * int(pct / 1.5)
    print(f"    {h:02d}:xx  {n:5,} ({pct:4.1f}%) {bar}")

print(f"\n--- Late OOS: Stable vs Unstable Lambda Bars ---")
oos_p33 = reg_oos['lambda_window_std'].quantile(0.33)
oos_p67 = reg_oos['lambda_window_std'].quantile(0.67)

reg_oos_late_stable   = reg_oos.loc[
    late_mask & (reg_oos['lambda_window_std'] <= oos_p33)
].dropna(subset=REGRESSION_COLS)
reg_oos_late_unstable = reg_oos.loc[
    late_mask & (reg_oos['lambda_window_std'] >  oos_p67)
].dropna(subset=REGRESSION_COLS)

for label, subset in [('Late OOS — stable lambda',   reg_oos_late_stable),
                       ('Late OOS — unstable lambda', reg_oos_late_unstable)]:
    if len(subset) < 100:
        print(f"\n  {label}: N={len(subset)} — too few observations, skipping")
        continue
    m = _fit_ols(subset['fwd_return'], PRIMARY_COLS, subset)
    print(f"\n  {label} | N={int(m.nobs):,}")
    print(f"    β₃ (tfi_x_regime): {m.params['tfi_x_regime']:.6f}  "
          f"p={m.pvalues['tfi_x_regime']:.4f}")

# ── Afternoon subsample ───────────────────────────────────────────────────────
print(f"\n--- Afternoon Subsample (13:xx–14:xx) ---")
reg_afternoon = reg[reg.index.hour.isin([13, 14])].dropna(subset=REGRESSION_COLS)
model_aft     = _fit_ols(reg_afternoon['fwd_return'], PRIMARY_COLS, reg_afternoon)

print(f"  N={int(model_aft.nobs):,}")
print(f"    β₃ (tfi_x_regime): {model_aft.params['tfi_x_regime']:.6f}")
print(f"    z-stat:            {model_aft.tvalues['tfi_x_regime']:.3f}")
print(f"    p-value:           {model_aft.pvalues['tfi_x_regime']:.4f}")
print(f"    β₁ (tfi):          {model_aft.params['tfi']:.6f}")
print(f"    R²:                {model_aft.rsquared:.6f}")

# =============================================================================
# OOS LAMBDA LEVEL COMPARISON
# =============================================================================
# Uses RegimeScore as a proxy for average lambda level — tests whether OOS
# lambda was structurally elevated relative to in-sample, which could
# mechanically inflate the OOS interaction term via circularity.

print("\n" + "=" * 60)
print("OOS LAMBDA LEVEL COMPARISON")
print("=" * 60)
print("  RegimeScore percentile comparison (>1.10x OOS/IS ratio would")
print("  indicate structurally elevated lambda as a concern).")

percentiles = [
    ('Mean', reg['regime_score'].mean(),         reg_oos['regime_score'].mean()),
    ('75th', reg['regime_score'].quantile(0.75), reg_oos['regime_score'].quantile(0.75)),
    ('90th', reg['regime_score'].quantile(0.90), reg_oos['regime_score'].quantile(0.90)),
]

print(f"\n  {'Percentile':<15} {'In-sample':>12} {'OOS':>12} {'OOS/IS ratio':>14}")
print(f"  {'-' * 55}")
for label, is_val, oos_val in percentiles:
    ratio = oos_val / is_val if is_val > 0 else float('nan')
    print(f"  {label:<15} {is_val:>12.4f} {oos_val:>12.4f} {ratio:>14.3f}x")

# =============================================================================
# 11. LAGGED REGIME CONDITIONING — RegimeScore_{t-1} on Return_{t+1}
# =============================================================================
# The cleanest possible test: fully predetermined regime score with zero
# simultaneity between any variable. Tests whether prior-bar regime
# information has any carry-forward predictive value for next-bar returns.

print("\n" + "=" * 60)
print("LAGGED REGIME CONDITIONING — RegimeScore_{t-1} on Return_{t+1}")
print("=" * 60)

model_lag = _fit_ols(reg['fwd_return'],
                     ['tfi', 'regime_score_lag', 'tfi_x_regime_lag', 'lag_return', 'lag_tfi'],
                     reg)

print(f"\n  Lagged-regime Primary | N={int(model_lag.nobs):,}")
print(f"    β₃ (tfi_x_regime_lag): {model_lag.params['tfi_x_regime_lag']:.6f}")
print(f"    z-stat:                {model_lag.tvalues['tfi_x_regime_lag']:.3f}")
print(f"    p-value:               {model_lag.pvalues['tfi_x_regime_lag']:.4f}")

# =============================================================================
# 12. MIDDAY SUBSAMPLE — 11:00–13:00 ET
# =============================================================================
# Pre-specified time window: lambda estimation is most stable (full 30-bar
# window of same-session data), TAR z-score most informative (lowest baseline
# activity), and structural contamination from open/close dynamics is absent.

print("\n" + "=" * 60)
print("MIDDAY SUBSAMPLE — 11:00–13:00 ET (hour ∈ {11, 12})")
print("=" * 60)

reg_midday   = reg[reg.index.hour.isin([11, 12])].dropna(subset=REGRESSION_COLS)
model_midday = _fit_ols(reg_midday['fwd_return'], PRIMARY_COLS, reg_midday)

print(f"\n  Midday Primary | N={int(model_midday.nobs):,}")
print(f"    β₃ (tfi_x_regime): {model_midday.params['tfi_x_regime']:.6f}")
print(f"    z-stat:            {model_midday.tvalues['tfi_x_regime']:.3f}")
print(f"    p-value:           {model_midday.pvalues['tfi_x_regime']:.4f}")

print(f"\n  β₃ summary across time windows:")
print(f"    Full sample:       β₃={model.params['tfi_x_regime']:.6f}  "
      f"p={model.pvalues['tfi_x_regime']:.4f}")
print(f"    Midday (11–12):    β₃={model_midday.params['tfi_x_regime']:.6f}  "
      f"p={model_midday.pvalues['tfi_x_regime']:.4f}")
print(f"    Afternoon (13–14): β₃={model_aft.params['tfi_x_regime']:.6f}  "
      f"p={model_aft.pvalues['tfi_x_regime']:.4f}")
print(f"    Stable lambda:     β₃={model_stable.params['tfi_x_regime']:.6f}  "
      f"p={model_stable.pvalues['tfi_x_regime']:.4f}")

# =============================================================================
# 13. TFI QUINTILE INTERACTION — Bonferroni-Corrected
# =============================================================================
# Replaces the continuous regime interaction term with four quintile dummy
# interactions (Q3 omitted as reference). Two purposes:
#   (1) Robustness: confirms the T+1 null is not an artifact of forcing
#       linearity on the regime-TFI interaction.
#   (2) Circularity evidence: a monotonic Q1→Q5 pattern indicates
#       mechanical co-elevation of TFI and RegimeScore rather than genuine
#       informed trading concentration in moderate quintiles (Kyle, 1985).

print("\n" + "=" * 60)
print("TFI QUINTILE INTERACTION — Bonferroni α=0.01 (5 tests)")
print("=" * 60)

reg['tfi_quintile'] = pd.qcut(reg['tfi'], q=5, labels=[1, 2, 3, 4, 5])
quintile_cols       = []
for q in [1, 2, 4, 5]:
    col = f'tfi_q{q}_x_regime'
    reg[col] = (reg['tfi_quintile'] == q).astype(float) * reg['regime_score']
    quintile_cols.append(col)

model_quintile = _fit_ols(
    reg['fwd_return'],
    ['tfi', 'regime_score'] + quintile_cols + ['lag_return', 'lag_tfi'],
    reg,
)

print(f"\n  Quintile Interaction | N={int(model_quintile.nobs):,}  (reference: Q3)")
print(f"  {'Term':<22} {'Coeff':>12} {'z-stat':>8} {'p-value':>10} {'Bonferroni':>12}")
print(f"  {'-' * 68}")
for col in quintile_cols:
    c        = model_quintile.params[col]
    t        = model_quintile.tvalues[col]
    p        = model_quintile.pvalues[col]
    survives = 'SURVIVES' if p < BONFERRONI_ALPHA else '—'
    print(f"  {col:<22} {c:>12.6f} {t:>8.3f} {p:>10.4f} {survives:>12}")

surviving = [c for c in quintile_cols if model_quintile.pvalues[c] < BONFERRONI_ALPHA]
print(f"\n  Bonferroni threshold (α/5): {BONFERRONI_ALPHA:.4f}")
if surviving:
    print(f"  Quintiles surviving correction: {', '.join(surviving)}")
else:
    print(f"  No quintile interactions survive Bonferroni correction.")

# =============================================================================
# 14. REGIME TRANSITION DYNAMICS
# =============================================================================
# Tests whether TFI is more predictive at the first bar of a low-to-high
# regime transition than during sustained high-regime periods.
# TransitionToHigh_t = 1 when RegimeScore crosses 0.5 from below, AND the
# prior bar was not in an announcement exclusion window (to prevent false
# transitions from exclusion resets inflating the transition flag).

print("\n" + "=" * 60)
print("REGIME TRANSITION DYNAMICS — Sustained vs Transition Effects")
print("=" * 60)

exclusion_mask_reg = exclusion_mask.reindex(reg.index).fillna(False).astype(bool)
prior_excluded     = exclusion_mask_reg.shift(1).fillna(False).astype(bool)

high = (reg['regime_score'] > 0.5).astype(int)
reg['transition_to_high'] = (
    (high == 1) & (high.shift(1) == 0) & (~prior_excluded)
).astype(float)
reg['tfi_x_transition'] = reg['tfi'] * reg['transition_to_high']

model_transition = _fit_ols(
    reg['fwd_return'],
    ['tfi', 'regime_score', 'tfi_x_regime', 'tfi_x_transition', 'lag_return', 'lag_tfi'],
    reg,
)

n_transitions = int(reg['transition_to_high'].sum())
print(f"\n  Transition Dynamics | N={int(model_transition.nobs):,}  "
      f"({n_transitions:,} transition bars, "
      f"{n_transitions / n_final * 100:.1f}% of sample)")
print(f"    Sustained β₃ (tfi_x_regime):     "
      f"{model_transition.params['tfi_x_regime']:.6f}  "
      f"p={model_transition.pvalues['tfi_x_regime']:.4f}")
print(f"    Transition β  (tfi_x_transition): "
      f"{model_transition.params['tfi_x_transition']:.6f}  "
      f"p={model_transition.pvalues['tfi_x_transition']:.4f}")

# =============================================================================
# 15. TRANSITION DYNAMICS — DELTA MAGNITUDE DIAGNOSTIC
# =============================================================================
# Tests the circularity explanation for the transition finding. If the
# transition coefficient is driven by mechanical inflation (large RegimeScore
# delta at crossing → large interaction term), it should concentrate in
# large-delta transitions. Under genuine informed trader timing, the effect
# should be consistent regardless of how large the delta is at the crossing.

print("\n" + "=" * 60)
print("TRANSITION DYNAMICS — DELTA MAGNITUDE DIAGNOSTIC")
print("=" * 60)
print("  Circularity predicts: large-delta β >> small-delta β")
print("  Genuine signal predicts: β similar regardless of delta size")

reg['regime_score_prev'] = reg['regime_score'].shift(1)
reg['transition_delta']  = np.where(
    reg['transition_to_high'] == 1,
    reg['regime_score'] - reg['regime_score_prev'],
    np.nan,
)

transition_bars = reg[reg['transition_to_high'] == 1].dropna(subset=['transition_delta'])

if len(transition_bars) < 50:
    print(f"  Insufficient transition bars ({len(transition_bars)}) — skipping.")
else:
    delta_median = transition_bars['transition_delta'].median()
    print(f"\n  Transition bars: {len(transition_bars):,}  "
          f"(median delta = {delta_median:.4f},  "
          f"mean = {transition_bars['transition_delta'].mean():.4f},  "
          f"std = {transition_bars['transition_delta'].std():.4f})")

    for split_label, split_cond in [
        ('Small delta (≤ median)', transition_bars['transition_delta'] <= delta_median),
        ('Large delta (> median)', transition_bars['transition_delta'] >  delta_median),
    ]:
        split_idx            = transition_bars[split_cond].index
        reg_d                = reg.copy()
        reg_d['split_flag']  = 0
        reg_d.loc[reg_d.index.isin(split_idx), 'split_flag'] = 1
        reg_d['tfi_x_split'] = reg_d['tfi'] * reg_d['split_flag']

        n_split = int(reg_d['split_flag'].sum())
        if n_split < 30:
            print(f"\n  {split_label}: only {n_split} bars — skipping.")
            continue

        X_d = sm.add_constant(
            reg_d[['tfi', 'regime_score', 'tfi_x_regime',
                    'tfi_x_split', 'lag_return', 'lag_tfi']]
        ).dropna()
        y_d     = reg_d['fwd_return'].reindex(X_d.index)
        model_d = sm.OLS(y_d, X_d).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

        print(f"\n  {split_label} | N transitions={n_split}")
        print(f"    Transition β: {model_d.params['tfi_x_split']:.6f}  "
              f"p={model_d.pvalues['tfi_x_split']:.4f}")

# =============================================================================
# 16. TRANSITION THRESHOLD ROBUSTNESS — 0.4 and 0.6
# =============================================================================
# Tests whether the transition result is stable across alternative crossing
# thresholds. If the result is threshold-specific, it is more likely an
# artifact of the mechanical inflation documented in Section 15.

print("\n" + "=" * 60)
print("TRANSITION THRESHOLD ROBUSTNESS — 0.4 and 0.6")
print("=" * 60)

exclusion_aligned = exclusion_mask.reindex(reg.index).fillna(False).astype(bool)
prior_excl_robust = exclusion_aligned.shift(1).fillna(False).astype(bool)

for threshold in [0.4, 0.6]:
    high_t    = (reg['regime_score'] > threshold).astype(int)
    trans_col = f'transition_to_high_{threshold}'
    inter_col = f'tfi_x_transition_{threshold}'

    reg[trans_col] = (
        (high_t == 1) & (high_t.shift(1) == 0) & (~prior_excl_robust)
    ).astype(float)
    reg[inter_col] = reg['tfi'] * reg[trans_col]

    n_trans = int(reg[trans_col].sum())
    m_tr    = _fit_ols(
        reg['fwd_return'],
        ['tfi', 'regime_score', 'tfi_x_regime', inter_col, 'lag_return', 'lag_tfi'],
        reg,
    )

    print(f"\n  Threshold = {threshold} | Transition bars: {n_trans:,}")
    print(f"    Sustained β₃ (tfi_x_regime):    "
          f"{m_tr.params['tfi_x_regime']:.6f}  p={m_tr.pvalues['tfi_x_regime']:.4f}")
    print(f"    Transition β  ({inter_col}): "
          f"{m_tr.params[inter_col]:.6f}  p={m_tr.pvalues[inter_col]:.4f}")

print(f"\n  Reference threshold=0.5 | Transition bars: {n_transitions:,}")
print(f"    Sustained β₃: {model_transition.params['tfi_x_regime']:.6f}  "
      f"p={model_transition.pvalues['tfi_x_regime']:.4f}")
print(f"    Transition β: {model_transition.params['tfi_x_transition']:.6f}  "
      f"p={model_transition.pvalues['tfi_x_transition']:.4f}")

# =============================================================================
# 17. TRANSACTION COST ANALYSIS
# =============================================================================
# Computes one-way and round-trip costs for ES futures (one tick = $12.50
# per contract, tick size = 0.25 index points). Evaluates whether predicted
# gross returns from either regression survive realistic transaction costs.

print("\n" + "=" * 60)
print("TRANSACTION COST ANALYSIS")
print("=" * 60)

mean_price     = df_indexed['price'].resample('1min').last().dropna().mean()
one_tick_log   = float(np.log1p(ES_TICK / mean_price))
round_trip_log = 2.0 * one_tick_log

print(f"\n  Mean ES close price:         {mean_price:,.2f}")
print(f"  One-way cost (1 tick):       {one_tick_log * 1e4:.3f} bps")
print(f"  Round-trip cost (2 ticks):   {round_trip_log * 1e4:.3f} bps")

mean_lag_ret  = float(reg['lag_return'].mean())
mean_lag_tfi  = float(reg['lag_tfi'].mean())
high_mask     = reg['regime_score'] > 0.5
mean_regime_h = float(reg.loc[high_mask, 'regime_score'].mean())
tfi_p25       = float(reg.loc[high_mask, 'tfi'].quantile(0.25))
tfi_p50       = float(reg.loc[high_mask, 'tfi'].median())
tfi_p75       = float(reg.loc[high_mask, 'tfi'].quantile(0.75))

print(f"\n  High-regime TFI percentiles (RegimeScore > 0.5):")
print(f"    25th: {tfi_p25:.4f}   Median: {tfi_p50:.4f}   75th: {tfi_p75:.4f}")

# ── Contemporaneous spec ──────────────────────────────────────────────────────
a_c  = model_contemp.params['const']
b1_c = model_contemp.params['tfi']
b2_c = model_contemp.params['regime_score_lag']
b3_c = model_contemp.params['tfi_x_regime_lag']
b5_c = model_contemp.params['lag_tfi']
total_c   = b1_c + b3_c
p_contemp = model_contemp.pvalues['tfi_x_regime_lag']

def _pred_contemp(tfi_val, regime_val):
    """Predicted return for contemporaneous spec at given TFI and regime values."""
    return (a_c + b1_c * tfi_val + b2_c * regime_val
            + b3_c * tfi_val * regime_val + b5_c * mean_lag_tfi)

print(f"\n  --- Contemporaneous regression (β₃={b3_c:.6f}, p={p_contemp:.4f}) ---")
print(f"  {'TFI':<12} {'Gross (bps)':>12} {'Net 1-way (bps)':>16} {'Net RT (bps)':>14}")
print(f"  {'-' * 56}")
for tfi_val, tag in [(tfi_p25, 'p25 (short)'), (tfi_p50, 'p50 (mid)  '), (tfi_p75, 'p75 (long) ')]:
    gross = _pred_contemp(tfi_val, mean_regime_h)
    net1  = gross - one_tick_log   * np.sign(tfi_val)
    net2  = gross - round_trip_log * np.sign(tfi_val)
    print(f"  {tag:<12} {gross * 1e4:>12.3f} {net1 * 1e4:>16.3f} {net2 * 1e4:>14.3f}")

if abs(total_c) > 1e-12:
    tfi_be_1way = (one_tick_log   - a_c - b2_c * mean_regime_h - b5_c * mean_lag_tfi) / total_c
    tfi_be_rt   = (round_trip_log - a_c - b2_c * mean_regime_h - b5_c * mean_lag_tfi) / total_c
    print(f"\n  Break-even TFI (RegimeScore={mean_regime_h:.3f}):")
    print(f"    One-way  (1 tick):    {tfi_be_1way:.4f}")
    print(f"    Round-trip (2 ticks): {tfi_be_rt:.4f}")

# ── T+1 spec ──────────────────────────────────────────────────────────────────
a_p  = model.params['const']
b1_p = model.params['tfi']
b2_p = model.params['regime_score']
b3_p = model.params['tfi_x_regime']
b4_p = model.params['lag_return']
b5_p = model.params['lag_tfi']
total_p   = b1_p + b3_p
p_primary = model.pvalues['tfi_x_regime']

def _pred_primary(tfi_val, regime_val):
    """Predicted return for T+1 spec at given TFI and regime values."""
    return (a_p + b1_p * tfi_val + b2_p * regime_val + b3_p * tfi_val * regime_val
            + b4_p * mean_lag_ret + b5_p * mean_lag_tfi)

print(f"\n  --- T+1 regression (β₃={b3_p:.6f}, p={p_primary:.4f}) ---")
print(f"  {'TFI':<12} {'Gross (bps)':>12} {'Net 1-way (bps)':>16} {'Net RT (bps)':>14}")
print(f"  {'-' * 56}")
for tfi_val, tag in [(tfi_p25, 'p25 (short)'), (tfi_p50, 'p50 (mid)  '), (tfi_p75, 'p75 (long) ')]:
    gross = _pred_primary(tfi_val, mean_regime_h)
    net1  = gross - one_tick_log   * np.sign(tfi_val)
    net2  = gross - round_trip_log * np.sign(tfi_val)
    print(f"  {tag:<12} {gross * 1e4:>12.3f} {net1 * 1e4:>16.3f} {net2 * 1e4:>14.3f}")

# =============================================================================
# 18. SAVE RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("SAVING RESULTS TO results/phase4/")
print("=" * 60)

_save_model(model,            'primary_regression.txt')
_save_model(model_contemp,    'contemporaneous_regression.txt')
_save_model(model_oos,        'oos_primary_regression.txt')
_save_model(model_lag,        'lagged_regime_regression.txt')
_save_model(model_midday,     'midday_regression.txt')
_save_model(model_quintile,   'quintile_interaction_regression.txt')
_save_model(model_transition, 'transition_dynamics_regression.txt')

for h, m in horizon_models.items():
    _save_model(m, f'horizon_t{h}_regression.txt')
for key, m in subsample_models.items():
    _save_model(m, f'subsample_{key}_regression.txt')

# Collate key coefficients across all models into a single CSV
rows = []
_collect_rows(model,            'primary_t1',          rows)
_collect_rows(model_contemp,    'contemporaneous',     rows)
_collect_rows(model_oos,        'oos_primary_t1',      rows)
_collect_rows(model_lag,        'lagged_regime_t1',    rows)
_collect_rows(model_midday,     'midday_t1',           rows)
_collect_rows(model_transition, 'transition_dynamics', rows)
for h, m in horizon_models.items():
    _collect_rows(m, f'horizon_t{h}', rows)
for key, m in subsample_models.items():
    _collect_rows(m, f'subsample_{key}', rows)

pd.DataFrame(rows).to_csv(
    os.path.join(RESULTS_DIR, 'key_coefficients.csv'),
    index=False,
    float_format='%.8f',
)
print(f"  Saved: key_coefficients.csv")

# Transaction cost summary file
tc_path = os.path.join(RESULTS_DIR, 'transaction_cost_analysis.txt')
with open(tc_path, 'w') as f:
    f.write("TRANSACTION COST ANALYSIS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Mean ES close price:          {mean_price:,.2f}\n")
    f.write(f"One-way cost (1 tick):        {one_tick_log * 1e4:.4f} bps\n")
    f.write(f"Round-trip cost (2 ticks):    {round_trip_log * 1e4:.4f} bps\n\n")
    f.write("CONTEMPORANEOUS REGRESSION:\n")
    f.write(f"  beta_TFI:                   {b1_c:.8f}\n")
    f.write(f"  beta_interaction:           {b3_c:.8f}\n")
    f.write(f"  beta_TFI + beta_int:        {total_c:.8f}\n")
    if abs(total_c) > 1e-12:
        f.write(f"  Break-even TFI (1-way):    {tfi_be_1way:.6f}\n")
        f.write(f"  Break-even TFI (RT):       {tfi_be_rt:.6f}\n")
    f.write("\nT+1 REGRESSION:\n")
    f.write(f"  beta_TFI:                   {b1_p:.8f}\n")
    f.write(f"  beta_interaction:           {b3_p:.8f}\n")
    f.write(f"  beta_TFI + beta_int:        {total_p:.8f}\n")
print(f"  Saved: transaction_cost_analysis.txt")

print("\n" + "=" * 60)
print("PHASE 4 ANALYSIS COMPLETE")
print("=" * 60)
