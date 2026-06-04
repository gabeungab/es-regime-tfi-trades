"""
test_signals.py — Signal construction sanity checks.

Loads the full in-sample ES futures trades dataset and verifies that
Kyle's lambda, trade arrival rate, the exclusion mask, and RegimeScore
are computed correctly. Prints summary statistics and breakdowns for
each signal to confirm expected behavior before running formal analysis.

Expected outputs (in-sample period):
    RegimeScore > 0.5 (high-regime): ~12.1% of valid bars
    Exclusion mask: ~15-20% of total bars excluded
    Kyle's lambda: positive mean, right-skewed
    Trade arrival rate: elevated at open and close
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_days, remove_outliers
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
TZ = 'America/New_York'

# Scheduled macro announcement datetimes (Eastern).
# FOMC, CPI, and NFP releases for in-sample period 2025-05-01 to 2025-12-30.
# Verified against official Fed/BLS release calendars.
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

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _print_series_stats(name, s):
    """Print summary statistics for a signal Series."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"  shape      : {s.shape}")
    print(f"  index range: {s.index.min()}  →  {s.index.max()}")
    print(f"  mean       : {s.mean():.6f}")
    print(f"  std        : {s.std():.6f}")
    print(f"  min        : {s.min():.6f}")
    print(f"  max        : {s.max():.6f}")
    print(f"  NaN count  : {s.isna().sum()}")


def _print_lambda_rth_check(lambda_series):
    """Print RTH-specific NaN breakdown for Kyle's lambda."""
    rth_mask = (
        (lambda_series.index.time >= pd.Timestamp('09:30').time()) &
        (lambda_series.index.time <= pd.Timestamp('16:00').time())
    )
    n_nan   = lambda_series[rth_mask].isna().sum()
    n_valid = lambda_series[rth_mask].notna().sum()
    print(f"  RTH NaN count  : {n_nan}")
    print(f"  RTH valid count: {n_valid}")
    print(f"  (NaNs in RTH arise from rolling warmup — expected for first "
          f"30 bars of each session)")


def _print_exclusion_breakdown(exclusion_mask):
    """Print exclusion mask summary."""
    n_total    = len(exclusion_mask)
    n_excluded = exclusion_mask.sum()
    n_retained = n_total - n_excluded
    print(f"\n{'=' * 60}")
    print(f"Exclusion mask")
    print(f"  total bars    : {n_total:,}")
    print(f"  excluded bars : {n_excluded:,}  "
          f"({n_excluded / n_total:.4f} of total)")
    print(f"  retained bars : {n_retained:,}")


def _print_regime_score_breakdown(regime_score):
    """Print RegimeScore threshold breakdown."""
    n_valid    = regime_score.notna().sum()
    n_above_50 = (regime_score > 0.50).sum()
    n_above_75 = (regime_score > 0.75).sum()
    n_zero     = (regime_score == 0.0).sum()
    print(f"\n  RegimeScore breakdowns (non-NaN bars = {n_valid:,}):")
    print(f"    fraction > 0.50 : {n_above_50 / n_valid:.4f}  "
          f"({n_above_50:,} bars)  ← high-regime threshold")
    print(f"    fraction > 0.75 : {n_above_75 / n_valid:.4f}  "
          f"({n_above_75:,} bars)")
    print(f"    fraction = 0.00 : {n_zero / n_valid:.4f}  "
          f"({n_zero:,} bars, set by exclusion mask)")
    print(f"\n  Expected high-regime fraction: ~12.1% (multiplicative detector)")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ── Load and clean data ───────────────────────────────────────────────────

    print("Loading in-sample data...")
    df       = load_all_days(DATA_DIR)
    df_clean = remove_outliers(df)
    print(f"    {len(df_clean):,} clean RTH trades across "
          f"{df_clean['ts_event_et'].dt.date.nunique()} trading days")

    # ── Compute signals ───────────────────────────────────────────────────────

    print("\nComputing signals...")
    lambda_series  = compute_lambda(df_clean)
    arrival_series = compute_arrival_rate(df_clean)

    df_indexed = df_clean.set_index('ts_event_et')
    bars       = df_indexed['price'].resample('1min').count()

    # Strip timezone from announcement dates if bar index is tz-naive
    ann_dates = ANNOUNCEMENT_DATES
    if bars.index.tzinfo is None:
        ann_dates = [dt.tz_localize(None) for dt in ann_dates]

    exclusion_mask = compute_exclusion_mask(bars, ann_dates)
    regime_score   = compute_regime_score(
        lambda_series, arrival_series, exclusion_mask
    )

    # ── Print sanity checks ───────────────────────────────────────────────────

    _print_series_stats("Kyle's lambda", lambda_series)
    _print_lambda_rth_check(lambda_series)

    _print_series_stats("Trade arrival rate", arrival_series)

    _print_exclusion_breakdown(exclusion_mask)

    _print_series_stats("RegimeScore", regime_score)
    _print_regime_score_breakdown(regime_score)

    print(f"\n{'=' * 60}")
    print("Signal checks complete.")
