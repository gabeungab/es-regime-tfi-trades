"""
signal_construction.py — Signal and regime score construction.

Provides four functions used throughout the analysis pipeline:

    compute_lambda          — Kyle's lambda (rolling price impact estimate)
    compute_arrival_rate    — Trade arrival rate (rolling mean trade count)
    compute_regime_score    — Multiplicative informed trading regime score
    compute_exclusion_mask  — Boolean mask for contaminated bars

All functions operate on 1-minute bar resolution and assume RTH-only
input data (9:30 AM – 4:00 PM ET). Rolling estimates use only past
data — no lookahead bias is introduced.
"""

import pandas as pd
import numpy as np


# Contract roll dates for ES futures within the analysis period.
# Each entry is the first day of the new front-month contract.
# Used by compute_exclusion_mask to exclude roll dates and the
# three preceding trading days.
ROLL_DATES = [
    pd.Timestamp('2025-06-19'),  # ESM5 → ESU5
    pd.Timestamp('2025-09-18'),  # ESU5 → ESZ5
    pd.Timestamp('2025-12-18'),  # ESZ5 → ESH6
    pd.Timestamp('2026-03-19'),  # ESH6 → ESM6
]


def compute_lambda(df, window=30):
    """
    Estimate Kyle's lambda via rolling OLS of 1-minute bar price changes
    on aggregate signed order flow. Lambda measures price impact per unit
    of order flow and serves as the primary proxy for information asymmetry.

    Estimation uses the closed-form OLS coefficient:
        lambda = cov(delta_p, signed_flow) / var(signed_flow)

    computed over a rolling window of 1-minute bars. NaN is returned for
    bars where the rolling window is incomplete (first `window` bars of
    each session) or where signed flow variance is zero.

    Parameters
    ----------
    df : pd.DataFrame
        Clean RTH trades DataFrame with ts_event_et, price, size,
        and side columns.
    window : int
        Rolling window in 1-minute bars. Default 30.

    Returns
    -------
    pd.Series
        Kyle's lambda estimate at each 1-minute bar, indexed by
        ts_event_et. Positive values indicate upward price impact
        from net buying pressure.
    """
    df = df.set_index('ts_event_et')

    signed_flow = (
        df['size'] * df['side'].map({'B': 1, 'A': -1, 'N': 0}).fillna(0)
    )

    closes   = df['price'].resample('1min').last()
    bar_dp   = closes.diff()
    bar_flow = signed_flow.resample('1min').sum()

    roll_cov = bar_dp.rolling(window=window, min_periods=window).cov(bar_flow)
    roll_var = bar_flow.rolling(window=window, min_periods=window).var()

    lambda_series = roll_cov / roll_var.replace(0, float('nan'))

    return lambda_series


def compute_arrival_rate(df, window=5):
    """
    Estimate the trade arrival rate as a rolling mean of 1-minute trade counts.
    Elevated arrival rates indicate heightened market activity and serve as a
    depth confirmation for elevated Kyle's lambda — ruling out thin markets
    or structural illiquidity as the source of elevated price impact.

    Parameters
    ----------
    df : pd.DataFrame
        Clean RTH trades DataFrame with ts_event_et and price columns.
    window : int
        Rolling window in 1-minute bars. Default 5. A shorter window
        than lambda is used to reflect the faster response of trading
        activity to changing market conditions.

    Returns
    -------
    pd.Series
        Rolling mean trades-per-minute at each 1-minute bar, indexed
        by ts_event_et.
    """
    df = df.set_index('ts_event_et')

    # Count trades per bar using price column; any non-null column works here.
    trade_count    = df['price'].resample('1min').count()
    arrival_series = trade_count.rolling(window=window, min_periods=window).mean()

    return arrival_series


def compute_regime_score(lambda_series, arrival_series, exclusion_mask,
                         lambda_window=30, arrival_window=5):
    """
    Combine Kyle's lambda and trade arrival rate into a continuous RegimeScore
    in [0, 1] using a multiplicative formulation:

        RegimeScore_t = logistic(z_lambda_t) × logistic(z_arrival_t)

    where logistic(x) = 1 / (1 + exp(-x)) and z_lambda, z_arrival are
    rolling z-scores of each component over their respective windows.

    The multiplicative structure enforces a hierarchical relationship:
    lambda provides the primary adverse selection signal, and TAR acts
    as a scaling factor. When lambda is low, RegimeScore is low regardless
    of TAR. When TAR is low, it suppresses the lambda signal proportionally,
    preventing misclassification of illiquid or thin-market episodes as
    informed trading regimes. RegimeScore is high only when both components
    are simultaneously elevated.

    Bars in the exclusion mask (post-announcement windows, MOC close period,
    contract roll windows) are set to 0.0 regardless of indicator values.

    Parameters
    ----------
    lambda_series : pd.Series
        Kyle's lambda estimates at each 1-minute bar.
    arrival_series : pd.Series
        Trade arrival rate estimates at each 1-minute bar.
    exclusion_mask : pd.Series
        Boolean Series indexed by ts_event_et; True where bar is excluded.
    lambda_window : int
        Rolling window for lambda z-score standardization. Default 30.
    arrival_window : int
        Rolling window for arrival rate z-score standardization. Default 5.

    Returns
    -------
    pd.Series
        RegimeScore in [0, 1] at each 1-minute bar, indexed by ts_event_et.
        Excluded bars are set to 0.0. NaN where rolling warmup is incomplete.
    """
    def _rolling_zscore(series, window):
        mean = series.rolling(window=window, min_periods=window).mean()
        std  = series.rolling(window=window, min_periods=window).std()
        return (series - mean) / std.replace(0, float('nan'))

    def _logistic(z):
        return 1 / (1 + np.exp(-z))

    z_lambda  = _rolling_zscore(lambda_series,  window=lambda_window)
    z_arrival = _rolling_zscore(arrival_series, window=arrival_window)

    # Multiplicative combination: each component mapped to (0, 1) independently
    # via logistic, then multiplied. TAR scales the lambda signal rather than
    # contributing additively — enforcing the intended hierarchy.
    regime_score = _logistic(z_lambda) * _logistic(z_arrival)

    # Zero out excluded bars after computing the score to avoid NaN propagation
    # from exclusion boundaries affecting adjacent valid bars.
    exclusion_aligned = exclusion_mask.reindex(regime_score.index, fill_value=False)
    regime_score      = regime_score.where(~exclusion_aligned, other=0.0)

    return regime_score


def compute_exclusion_mask(bars, announcement_dates, premarket_rth_shift=False):
    """
    Build a boolean exclusion mask over the 1-minute bar index, marking
    three categories of contaminated observations:

        1. Final 10 minutes of each RTH session (15:50–16:00 ET):
           excluded due to MOC order flow domination — mechanical,
           publicly announced execution that elevates volume and lambda
           without genuine adverse selection.

        2. Post-announcement windows (+30 minutes after each scheduled
           macro release): excluded because public information arrival
           drives directional flow that mechanically inflates lambda
           without reflecting private information. Pre-announcement
           windows are retained as potentially informed.

           For intraday releases (e.g., FOMC at 14:00): the window is
           [ann_dt, ann_dt + 30min], which falls within RTH.

           For pre-market releases (e.g., CPI/NFP/PPI at 08:30): the
           default behavior [ann_dt, ann_dt + 30min] produces a window
           of 08:30–09:00, which falls entirely before RTH and excludes
           zero bars. When premarket_rth_shift=True, the window is
           shifted to the first 30 minutes of RTH on the announcement
           day (09:30–10:00), capturing the opening period during which
           the market is reacting to the pre-open release.

        3. Contract roll dates and 3 preceding trading days: excluded
           to remove structural illiquidity and mechanical volume
           migration accompanying front-month contract expiration.

    All exclusions are restricted to RTH bars only (9:30–16:00 ET).

    Parameters
    ----------
    bars : pd.DataFrame or pd.Series
        1-minute bar DataFrame/Series indexed by ts_event_et.
    announcement_dates : list of pd.Timestamp
        Announcement datetimes (Eastern). Each triggers a +30-minute
        post-announcement exclusion window.
    premarket_rth_shift : bool, optional
        If True, pre-market announcements (time < 09:30 ET) shift their
        exclusion window to the first 30 minutes of RTH on the
        announcement day (09:30–10:00) rather than the pre-RTH interval
        [ann_dt, ann_dt + 30min]. Default False preserves the original
        behavior for backward compatibility with formal_analysis.py.

    Returns
    -------
    pd.Series
        Boolean Series indexed by ts_event_et; True where bar is excluded.
    """
    index = bars.index
    mask  = pd.Series(False, index=index)

    rth_open  = pd.Timestamp('09:30').time()
    rth_close = pd.Timestamp('16:00').time()
    is_rth    = pd.Series(
        (index.time >= rth_open) & (index.time <= rth_close),
        index=index
    )

    # ── 1. Final 10 minutes of each RTH session (MOC flow) ───────────────────
    cutoff_time = pd.Timestamp('15:50').time()
    mask |= (is_rth & pd.Series(index.time >= cutoff_time, index=index))

    # ── 2. Post-announcement windows (+30 min) ────────────────────────────────
    _bar_dates  = pd.Series(index.date, index=index)
    _rth_30_end = pd.Timestamp('10:00').time()

    for ann_dt in announcement_dates:
        if premarket_rth_shift and ann_dt.time() < rth_open:
            # Pre-market release: shift window to first 30 min of RTH on
            # the announcement day (09:30–10:00) rather than 08:30–09:00.
            ann_date = ann_dt.date()
            mask |= (
                (_bar_dates == ann_date) &
                pd.Series(index.time >= rth_open,    index=index) &
                pd.Series(index.time <  _rth_30_end, index=index)
            )
        else:
            window_start = ann_dt
            window_end   = ann_dt + pd.Timedelta(minutes=30)
            mask |= (
                (index >= window_start) &
                (index <= window_end) &
                is_rth
            )

    # ── 3. Contract roll dates and 3 preceding trading days ───────────────────
    index_dates = pd.Series(index.date, index=index)
    for roll_date in ROLL_DATES:
        roll_start     = (roll_date - pd.offsets.BDay(3)).date()
        in_roll_window = (
            (index_dates >= roll_start) &
            (index_dates <= roll_date.date())
        )
        mask |= (in_roll_window & is_rth)

    return mask
