"""
plot_signals.py — Phase 3 exploratory signal plots.

Generates four-panel diagnostic plots for each signal component
(Kyle's lambda, trade arrival rate, RegimeScore) and three
conditional TFI plots stratified by regime state. All outputs
are saved to results/phase3/.

Each component produces four figures:
    {signal}_timeseries.png       — daily aggregated value over sample
    {signal}_intraday.png         — mean value at each 1-minute bar of day
    {signal}_distribution.png     — full RTH distribution
    {signal}_individual_days.png  — four representative trading days

TFI conditional plots:
    tfi_scatter_by_regime.png      — TFI vs forward return by regime
    tfi_distribution_by_regime.png — TFI density by regime
    tfi_acf_by_regime.png          — TFI autocorrelation by regime
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf

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
    '~/Desktop/X9 Pro/raw-market-data/es-futures/trades/GLBX-20250501-20251231/'
)
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'phase3')
TZ       = 'America/New_York'

_RTH_OPEN  = pd.Timestamp('09:30').time()
_RTH_CLOSE = pd.Timestamp('16:00').time()

# Four representative trading days for individual-day panels.
# Selected to illustrate high-volatility, low-volatility, FOMC announcement,
# and a normal session within the in-sample period.
INDIVIDUAL_DAYS = {
    'high_vol':     pd.Timestamp('2025-06-13', tz=TZ),
    'low_vol':      pd.Timestamp('2025-07-25', tz=TZ),
    'announcement': pd.Timestamp('2025-07-30', tz=TZ),
    'normal':       pd.Timestamp('2025-08-13', tz=TZ),
}

_DAY_LABELS = {
    'high_vol':     'High Volatility (2025-06-13)',
    'low_vol':      'Low Volatility (2025-07-25)',
    'announcement': 'FOMC Announcement (2025-07-30)',
    'normal':       'Normal Day (2025-08-13)',
}

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
# INTERNAL HELPERS
# =============================================================================

def _plot_component(series, name, filename_prefix, save_dir,
                    daily_agg='mean', dist_filter='none', ts_filter='none',
                    n_trading_days=None):
    """
    Four-panel diagnostic plot for a 1-minute bar signal Series.
    Saves each panel as a separate PNG to save_dir.

    Parameters
    ----------
    series : pd.Series
        Signal values indexed by ts_event_et.
    name : str
        Display name for axis labels and titles.
    filename_prefix : str
        Prefix for output filenames (e.g. 'lambda', 'arrival').
    save_dir : str
        Directory to save output PNGs.
    daily_agg : str
        Aggregation for the daily time series panel — 'mean' or 'max'.
    dist_filter : str
        Filter applied to RTH values in the distribution panel.
        'none'     — no additional filter.
        'positive' — keep only values > 0 (use for arrival and RegimeScore
                     where zeros are set by the exclusion mask).
    ts_filter : str
        Filter applied to daily aggregated values in the time series panel.
        'none' or 'positive' — same logic as dist_filter.
    n_trading_days : int or None
        Number of trading days in the sample, used for the intraday average
        title. Should be passed explicitly from the calling scope (computed
        from df_clean) to avoid miscounting from signal series index spans.
        If None, falls back to counting non-empty daily buckets in the series
        (less reliable — prefer passing explicitly).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Use explicitly passed day count if available; otherwise approximate
    # from the series index. The explicit count is always more accurate because
    # the signal series index can span more calendar days than trading days.
    if n_trading_days is None:
        n_trading_days = int(
            pd.Series(series.dropna().index.date).nunique()
        )

    # ── 1. Full sample time series (resampled to daily) ───────────────────────
    daily_series = getattr(series.dropna().resample('1D'), daily_agg)()
    daily_series = daily_series[daily_series.notna()]
    if ts_filter == 'positive':
        daily_series = daily_series[daily_series > 0]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily_series.index, daily_series.values, linewidth=1.2)
    ax.set_title(f'{name} — Daily {daily_agg.capitalize()} Across Sample')
    ax.set_xlabel('Date')
    ax.set_ylabel(name)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{filename_prefix}_timeseries.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename_prefix}_timeseries.png")

    # ── 2. Intraday average ───────────────────────────────────────────────────
    rth_mask     = (
        (series.index.time >= _RTH_OPEN) &
        (series.index.time <= _RTH_CLOSE)
    )
    rth          = series[rth_mask].dropna()
    intraday_avg = rth.groupby(rth.index.time).mean()

    n              = len(intraday_avg)
    tick_positions = [0, 60, 120, 180, 240, 300, min(350, n - 41), min(390, n - 1)]
    tick_labels    = ['9:30', '10:30', '11:30', '12:30',
                      '13:30', '14:30', '15:30', '16:00']

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(n), intraday_avg.values)
    ax.set_title(
        f'{name} — Intraday Average ({n_trading_days}-Day Mean at Each Minute)'
    )
    ax.set_xlabel('Time of Day')
    ax.set_ylabel(f'Mean {name}')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{filename_prefix}_intraday.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename_prefix}_intraday.png")

    # ── 3. Distribution (RTH bars only) ──────────────────────────────────────
    rth_values = series[rth_mask].dropna().values
    if dist_filter == 'positive':
        rth_values = rth_values[rth_values > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rth_values, bins=100, edgecolor='none')
    ax.set_title(f'{name} — Distribution (RTH Bars Only)')
    ax.set_xlabel(name)
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{filename_prefix}_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename_prefix}_distribution.png")

    # ── 4. Individual days (2×2 grid) ────────────────────────────────────────
    series_et   = series.copy()
    series_et.index = series.index.tz_convert(TZ).tz_localize(None)
    local_index = series_et.index

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (key, ts) in zip(axes.flatten(), INDIVIDUAL_DAYS.items()):
        day_date   = ts.date()
        day_series = series_et[local_index.date == day_date].dropna()
        if len(day_series) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
        else:
            ax.plot(day_series.index, day_series.values)
            day   = day_series.index[0].date()
            ticks = [pd.Timestamp(f'{day} {t}') for t in
                     ['09:30', '10:30', '11:30', '12:30',
                      '13:30', '14:30', '15:30', '16:00']]
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlim(
                pd.Timestamp(f'{day} 09:30').tz_localize(None),
                pd.Timestamp(f'{day} 16:00').tz_localize(None),
            )
        ax.set_title(f'{name} — {_DAY_LABELS[key]}')
        ax.set_xlabel('Time')
        ax.set_ylabel(name)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{filename_prefix}_individual_days.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename_prefix}_individual_days.png")

# =============================================================================
# PUBLIC PLOT FUNCTIONS
# =============================================================================

def plot_lambda(lambda_series, save_dir=SAVE_DIR, n_trading_days=None):
    """Four-panel diagnostic plot for Kyle's lambda."""
    _plot_component(lambda_series, "Kyle's Lambda", 'lambda',
                    save_dir, daily_agg='max',
                    dist_filter='none', ts_filter='none',
                    n_trading_days=n_trading_days)


def plot_arrival(arrival_series, save_dir=SAVE_DIR, n_trading_days=None):
    """Four-panel diagnostic plot for trade arrival rate."""
    _plot_component(arrival_series, 'Trade Arrival Rate', 'arrival',
                    save_dir, daily_agg='mean',
                    dist_filter='positive', ts_filter='positive',
                    n_trading_days=n_trading_days)


def plot_regime_score(regime_score, save_dir=SAVE_DIR, n_trading_days=None):
    """Four-panel diagnostic plot for RegimeScore."""
    _plot_component(regime_score, 'RegimeScore', 'regime_score',
                    save_dir, daily_agg='mean',
                    dist_filter='positive', ts_filter='positive',
                    n_trading_days=n_trading_days)


def plot_tfi_by_regime(tfi_df, returns_df, regime_score, save_dir=SAVE_DIR):
    """
    Three plots showing TFI conditional on RegimeScore.

    Plot 1 — Scatter: TFI vs 1-min forward log return, side-by-side
             for high (>0.5) and low (≤0.5) RegimeScore bars.
             Up to 5,000 points sampled per panel for visual clarity;
             regression line uses full data.
    Plot 2 — Overlapping density histograms of TFI for each regime.
    Plot 3 — ACF of TFI at lags 1–20 for each regime, with 95%
             confidence bands.
    """
    os.makedirs(save_dir, exist_ok=True)

    tfi = tfi_df['tfi'] if isinstance(tfi_df, pd.DataFrame) else tfi_df
    fwd_return = (
        returns_df['log_return'] if isinstance(returns_df, pd.DataFrame)
        else returns_df
    ).shift(-1)

    df = pd.DataFrame({
        'tfi':          tfi,
        'fwd_return':   fwd_return,
        'regime_score': regime_score,
    }).dropna()

    high = df[df['regime_score'] > 0.5]
    low  = df[df['regime_score'] <= 0.5]

    print(f"  Aligned bars: {len(df):,}  "
          f"(high-regime: {len(high):,}, low-regime: {len(low):,})")

    # ── Plot 1: TFI vs forward return scatter by regime ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, subset, label, color in [
        (axes[0], high, 'High RegimeScore (>0.5)',    'steelblue'),
        (axes[1], low,  'Low RegimeScore (\u22640.5)', 'tomato'),
    ]:
        plot_subset = subset.sample(min(5_000, len(subset)), random_state=42)
        ax.scatter(plot_subset['tfi'], plot_subset['fwd_return'],
                   alpha=0.15, s=4, color=color, rasterized=True)
        if len(subset) > 1:
            m, b  = np.polyfit(subset['tfi'], subset['fwd_return'], 1)
            x_rng = np.linspace(subset['tfi'].min(), subset['tfi'].max(), 200)
            ax.plot(x_rng, m * x_rng + b, color='black', linewidth=1.5,
                    label=f'slope = {m:.2e}')
            ax.legend(fontsize=9)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.6, linestyle='--')
        ax.set_title(label)
        ax.set_xlabel('TFI')
        ax.set_ylabel('1-min Forward Log Return')
    fig.suptitle('TFI vs Forward Return by RegimeScore', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tfi_scatter_by_regime.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved tfi_scatter_by_regime.png")

    # ── Plot 2: TFI distribution by regime ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(high['tfi'].values, bins=80, alpha=0.5, density=True,
            color='steelblue', label='High RegimeScore (>0.5)')
    ax.hist(low['tfi'].values,  bins=80, alpha=0.5, density=True,
            color='tomato',    label='Low RegimeScore (\u22640.5)')
    ax.set_title('TFI Distribution by RegimeScore')
    ax.set_xlabel('TFI')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tfi_distribution_by_regime.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved tfi_distribution_by_regime.png")

    # ── Plot 3: TFI ACF by regime ─────────────────────────────────────────────
    nlags    = 20
    lags     = np.arange(1, nlags + 1)
    high_acf = acf(high['tfi'].values, nlags=nlags, fft=True)[1:]
    low_acf  = acf(low['tfi'].values,  nlags=nlags, fft=True)[1:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, acf_vals, n, label, color in [
        (axes[0], high_acf, len(high), 'High RegimeScore (>0.5)',    'steelblue'),
        (axes[1], low_acf,  len(low),  'Low RegimeScore (\u22640.5)', 'tomato'),
    ]:
        ci = 1.96 / np.sqrt(n)
        ax.bar(lags, acf_vals, color=color, alpha=0.75, width=0.6)
        ax.axhline( ci, color='gray',  linestyle='--', linewidth=1)
        ax.axhline(-ci, color='gray',  linestyle='--', linewidth=1)
        ax.axhline(0,   color='black', linewidth=0.5)
        ax.set_title(f'TFI ACF — {label}')
        ax.set_xlabel('Lag (minutes)')
        ax.set_ylabel('Autocorrelation')
        ax.set_xticks(lags)
    fig.suptitle('TFI Autocorrelation Function by RegimeScore', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'tfi_acf_by_regime.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved tfi_acf_by_regime.png")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ── Load and clean data ───────────────────────────────────────────────────

    print("Loading data...")
    df       = load_all_days(DATA_DIR)
    df_clean = remove_outliers(df)

    # Compute trading day count from the authoritative source — the clean
    # DataFrame — rather than inferring from signal series index spans,
    # which can overcount due to calendar day boundaries.
    n_days = df_clean['ts_event_et'].dt.date.nunique()

    print(f"    {len(df_clean):,} clean RTH trades across "
          f"{n_days} trading days")

    # ── Compute signals ───────────────────────────────────────────────────────

    print("\nComputing signals...")
    lambda_series  = compute_lambda(df_clean)
    arrival_series = compute_arrival_rate(df_clean)

    df_indexed = df_clean.set_index('ts_event_et')
    bars       = df_indexed['price'].resample('1min').count()

    ann_dates = ANNOUNCEMENT_DATES
    if bars.index.tzinfo is None:
        ann_dates = [dt.tz_localize(None) for dt in ann_dates]

    exclusion_mask = compute_exclusion_mask(bars, ann_dates)
    regime_score   = compute_regime_score(
        lambda_series, arrival_series, exclusion_mask
    )

    print("\nComputing TFI and returns...")
    tfi     = compute_tfi(df_clean)
    returns = compute_returns(df_clean)

    # ── Generate plots ────────────────────────────────────────────────────────

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\nGenerating lambda plots...")
    plot_lambda(lambda_series, n_trading_days=n_days)

    print("\nGenerating arrival rate plots...")
    plot_arrival(arrival_series, n_trading_days=n_days)

    print("\nGenerating RegimeScore plots...")
    plot_regime_score(regime_score, n_trading_days=n_days)

    print("\nGenerating TFI conditional plots...")
    plot_tfi_by_regime(tfi, returns, regime_score)

    print(f"\nDone. All plots saved to {SAVE_DIR}")
