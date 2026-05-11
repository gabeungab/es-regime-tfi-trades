"""
test_load.py — Phase 1 data loading and descriptive statistics checks.

Loads the full in-sample ES futures trades dataset, removes outliers,
computes daily summary statistics, and produces Phase 1 exploratory
figures. Run this script to verify the data pipeline is working correctly
and to regenerate Phase 1 outputs.

Output files written to results/phase1/:
    daily_stats.csv
    intraday_trade_arrival.png
    intraday_volume.png
    intraday_volatility.png
    trade_size_distribution.png
    daily_volume_over_time.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all_days, remove_outliers, compute_daily_stats, plot_phase1

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR    = os.path.expanduser(
    '~/Desktop/X9 Pro/raw-market-data/es-futures/trades/GLBX-20250501-20251231/'
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase1')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ── Load and clean data ───────────────────────────────────────────────────

    print("Loading in-sample data...")
    df = load_all_days(DATA_DIR)

    print("Removing outliers...")
    df_clean = remove_outliers(df)

    print(f"\nClean dataset: {len(df_clean):,} trades across "
          f"{df_clean['ts_event_et'].dt.date.nunique()} trading days")

    # ── Compute daily statistics ──────────────────────────────────────────────

    print("\nComputing daily statistics...")
    daily = compute_daily_stats(df_clean)

    # ── Save outputs ──────────────────────────────────────────────────────────

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, 'daily_stats.csv')
    daily.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    print("\nGenerating Phase 1 plots...")
    plot_phase1(df_clean, daily, save_dir=RESULTS_DIR)

    print("\nPhase 1 complete.")
