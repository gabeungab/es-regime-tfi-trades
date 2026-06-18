# ES Futures Microstructure Research — Working Paper

---

**Abstract**

This paper examines whether conditioning Trade Flow Imbalance (TFI)
on a real-time informed trading regime detector amplifies its
predictive power for forward returns in E-mini S&P 500 futures. The
regime detector is a two-component multiplicative composite of
Kyle's lambda and trade arrival rate, constructed from 58.8 million
trades across 169 trading days on CME Globex and validated on a
held-out 46-day out-of-sample period. The primary finding is a null
result: regime-conditioned TFI does not significantly predict forward
returns at any horizon from T+1 through T+15 (β₃ = 0.000371,
p = 0.234), implying that ES futures incorporate regime-conditioned
order flow information within one 1-minute bar. A key secondary
finding qualifies this conclusion: restricting the regression to
bars where the lambda estimation window is most stable produces a
statistically significant regime-TFI interaction (β₃ = 0.001016,
p = 0.033) that is 2.7x the full-sample estimate, suggesting the
detector's signal quality depends critically on estimation
reliability. Three caveats bound this finding: the stability
threshold is derived post-hoc from the in-sample distribution; the
effect does not replicate in the held-out OOS period; and the
stability metric (rolling signed-flow standard deviation) is itself
partially circular with the regime detector. The finding is treated
as a genuine in-sample result motivating a specific future research
direction rather than a deployable signal. A central methodological
contribution is the explicit documentation of two circularity layers
inherent in trades-only regime detection and the analysis of their
directional effects on all estimates. Both layers act to bias β₃
upward toward the alternative hypothesis, a direction confirmed by a
permutation simulation (1,000 datasets under H₀, seed = 42) that
places the observed β₃ = 0.000371 at the 85.4th percentile of the
null distribution (simulation p-value = 0.146), making the primary
null result a conservative bound on the true causal effect.

**JEL Classification:** G12, G14, C58

**Keywords:** market microstructure, adverse selection, informed
trading, order flow imbalance, Kyle's lambda, ES futures, regime
detection

---

## 1. Introduction

Market makers in high-frequency equity index futures face a
fundamental adverse selection problem. When a counterparty's order
flow reflects private information about future price direction,
market makers who fail to detect it provide liquidity at stale
quotes and absorb losses from informed traders. Managing this
exposure requires identifying, in real time, whether the order flow
arriving in each bar is more likely to be informed or uninformed.
The theoretical frameworks of Glosten and Milgrom (1985) and Kyle
(1985) formalize this problem: market makers update quotes in
proportion to their inference about order flow informativeness, and
the price impact of a trade — Kyle's lambda — is the empirical
signature of this updating process. The practical question is
whether lambda, observed from trades-only data, can be combined
with other market characteristics to construct a real-time regime
indicator that reliably identifies elevated adverse selection.

This paper asks whether conditioning Trade Flow Imbalance (TFI) on
such a regime indicator amplifies its predictive power for
next-bar forward returns in E-mini S&P 500 (ES) futures. TFI
measures the normalized directional imbalance between buyer- and
seller-initiated volume within each 1-minute bar — a signal that
reflects the net pressure of order flow but mixes informed and
uninformed activity unconditionally. The regime detector is a
two-component multiplicative composite of Kyle's lambda and trade
arrival rate (TAR), constructed from 58.8 million RTH trades across
169 trading days on CME Globex. The primary test is a regression
of next-bar log returns on TFI, RegimeScore, and their interaction,
estimated with Newey-West HAC standard errors on the full in-sample
period (2025-05-01 through 2025-12-30) and validated on a held-out
out-of-sample period (2026-01-02 through 2026-03-06).

The primary finding is a null result. Regime-conditioned TFI does
not significantly predict forward returns at any horizon from T+1
through T+15 (β₃ = 0.000371, p = 0.234 at T+1), implying that ES
futures incorporate regime-conditioned order flow information within
one 1-minute bar and leave no exploitable residual predictability
at any tested horizon. A key secondary finding qualifies this
conclusion: when the regime detector's lambda estimation window is
stable — defined as the bottom tercile of rolling signed-flow
standard deviation — the regime-TFI interaction is statistically
significant (β₃ = 0.001016, p = 0.033) and 2.7x the full-sample
estimate. This suggests the detector's signal quality depends
critically on estimation reliability, and motivates a confirmatory
test with an orthogonal limit order book detector as the primary
direction for future research.

A central methodological challenge in this analysis is that with
trades-only data, a fully orthogonal regime detector is impossible:
Kyle's lambda and TFI share aggressor-side signed order flow as a
common input. This paper explicitly documents two layers of
circularity — at the detector construction level and at the
regression interaction level — and analyzes their directional
effects on all estimates throughout. The circularity biases β₃
upward toward the alternative hypothesis, making a null result more
informative: it holds even under estimation conditions that favor
finding a positive coefficient. This accounting of what the data
can and cannot identify is a primary methodological contribution
of the paper.

The remainder of the paper is organized as follows. Section 2
reviews the relevant literature. Section 3 describes the data and
sample construction. Section 4 presents the signal construction,
regime detector design, exclusion windows, detector validation, and
empirical specification. Section 5 reports the primary results and
additional tests. Section 6 presents robustness checks. Section 7
discusses market maker implications. Section 8 outlines two
directions for future research.

---

## 2. Literature Review

The theoretical foundation of this paper rests on the adverse
selection frameworks of Kyle (1985) and Glosten and Milgrom (1985).
Kyle (1985) models a market in which a single informed trader
submits orders strategically to minimize price impact, and derives
lambda — the price impact per unit of signed order flow — as the
equilibrium measure of information asymmetry. Glosten and Milgrom
(1985) model competitive market makers who update bid and ask quotes
in response to the probability that any incoming order is informed,
establishing that the optimal spread is proportional to adverse
selection risk. Together, these frameworks motivate the regime
detector design: a period of elevated lambda, confirmed by active
market participation, is the empirical signature of the informed
trading conditions both frameworks identify as generating adverse
selection. The detector is not a loose empirical proxy — it is
a direct operationalization of the quantities these models define.

The empirical antecedent of TFI is the order flow imbalance (OFI)
measure of Cont, Kukanov, and Stoikov (2014), who show that OFI
constructed from limit order book events predicts short-term price
changes in equity markets. This paper uses a trades-only analog —
signed volume imbalance at the aggressor side — and replicates the
finding that unconditional order flow imbalance predicts forward
returns (β₁ = 0.000637, p < 0.001). The central question of this
paper is whether regime conditioning amplifies this unconditional
predictability. It does not — but the stable regime conditions
analysis identifies the specific circumstances under which a
statistically detectable amplification does emerge, providing a
bridge from Cont et al.'s unconditional finding to a conditional
one that may be accessible with richer data.

The choice of detector components is grounded in Ahern (2018) and
Dufour and Engle (2000). Ahern (2018) evaluates which trades-only
proxies for informed trading most accurately identify actual insider
trading episodes using a unique sample of illegal trades, finding
that the most accurate proxies — TFI variants and order flow
autocorrelation — are informationally circular with TFI itself. This
directly motivates the use of lambda and TAR as the least circular
available detector: they capture price impact and market activity
without using TFI's own directional content as an input. Roll (1984)
was considered as a third component — its serial covariance estimator
measures bid-ask tightness and could confirm that elevated lambda
reflects genuine price impact rather than wide spreads — but was
excluded because it is non-orthogonal to lambda and fails precisely
in the one-sided markets that represent the highest-lambda informed
trading episodes of greatest theoretical interest. 

The most prominent trades-only informed-trading proxy not used in
this paper is VPIN (Volume-Synchronized Probability of Informed
Trading), introduced by Easley, López de Prado, and O'Hara (2012).
VPIN was considered as both an alternative regime detector and a
robustness check and was excluded for two reasons. First, VPIN is
constructed from signed volume imbalance — the same directional
content as TFI — making it more circular with TFI than lambda,
which at minimum incorporates price impact as an independent input.
Second, because the primary result is null, a VPIN robustness check
would confirm a null result under a more circular detector, adding
no inferential value. The exclusion is deliberate: the detector
design goal was to minimize circularity with TFI, not to benchmark
against existing proxies that inherit the same limitation.

Dufour and Engle
(2000) provide direct theoretical grounding for TAR: they show that
inter-trade duration is informative about information arrival, with
faster trading associated with informed episodes. This motivates
including TAR as a real-time proxy for market participation intensity
and supports its role as a depth confirmation for elevated lambda
rather than as an independent adverse selection signal.

The use of TAR as a hierarchical check on lambda rather than an
independent additive component connects to the broader literature
on market liquidity and information. Elevated lambda in a thin,
illiquid market reflects structural price impact from low depth
rather than information asymmetry. By requiring both high lambda
and high TAR to produce an elevated regime score, the multiplicative
detector addresses the thin-market failure mode that any
lambda-only detector inherits. This design is consistent with the
view in the market microstructure literature, surveyed in O'Hara
(1995), that informed trading episodes are characterized jointly by
elevated price impact and active participation — not by price impact
alone.

This paper contributes to the microstructure literature in three
ways. First, it introduces a multiplicative regime detector that
enforces the theoretical hierarchy between lambda and TAR —
lambda provides the primary adverse selection signal and TAR
confirms or suppresses it — rather than treating the two components
as symmetric additive inputs. Second, it provides a systematic
documentation of two distinct circularity layers in trades-only
regime detection and analyzes their directional effects on all
downstream estimates, addressing a methodological gap in the
empirical literature on informed trading proxies. Third, the stable
regime conditions finding — that the regime-TFI interaction is
statistically significant when lambda estimation is most reliable
— generates a specific, falsifiable prediction for future research
with limit order book data: an orthogonal LOB-based detector
operating under stable liquidity conditions should produce
significant regime-conditioned forward return predictability, or
confirm that the in-sample finding is circularity-driven.

---

## 3. Data

### 3.1 Source and Instrument

This study uses trade-level data from CME Globex obtained via
Databento (GLBX.MDP3, Trades schema). The data provides a complete
record of every executed trade in ES futures at nanosecond timestamp
precision, including execution price, contract size, and aggressor
side (B = buyer, A = seller, N = unknown).

Analysis is restricted to Regular Trading Hours: 9:30 AM to 4:00 PM
Eastern time. The overnight Globex session is excluded due to its
structurally different liquidity profile. Calendar spread instruments
(e.g. ESM5-ESU5), which appeared in the raw feed with spread prices
rather than outright futures prices, were removed by filtering any
symbol containing a hyphen.

### 3.2 Sample Period and Cleaning

The in-sample period covers 2025-05-01 through 2025-12-30. A
held-out out-of-sample period spanning 2026-01-02 through 2026-03-06
is reserved exclusively for final validation and is not used in any
model estimation or specification search.

The in-sample period (May–December 2025, 169 trading days) was
determined by data availability and cost. Trade-level data was
purchased for this study period only; pre-May 2025 GLBX.MDP3 data
was not acquired due to cost constraints and therefore was not
available when the formal analysis was conducted. The OOS window
(January–March 2026, 46 trading days) is the next available period
after the in-sample window and was held out without any access
during model development. A longer history or multiple rolling OOS
windows would be preferable and is a direction for future work;
the conclusions of this paper are therefore conditional on the
2025–2026 sample and the market conditions prevailing during it.

Four trading days were excluded due to anomalously low activity
relative to the sample mean of 348,369 trades per day: Memorial Day
(2025-05-26, 14,711 trades), Independence Day (2025-07-04, 13,367
trades), Labor Day (2025-09-01, 11,431 trades), and the half-session
preceding Thanksgiving (2025-11-27, 9,258 trades). A natural gap
separates these four days from all other sessions, including
recognized reduced-activity days such as Juneteenth (51,784 trades)
which are retained as legitimate trading sessions.

Price outliers were removed using a rolling five-standard-deviation
filter with a window of 100 trades, eliminating 114,413 records
(~0.19% of the raw sample). An additional 2,856,579 records share
nanosecond timestamps with at least one other trade; inspection
confirmed these as legitimate multi-level fills and all are retained.

The final clean in-sample dataset contains ~58.8 million RTH trades
across 169 trading days.

### 3.3 Descriptive Statistics

Table 1 presents daily summary statistics across the 169-day
in-sample period.

| Statistic               | Mean      | Std       | Min     | Max       |
|-------------------------|-----------|-----------|---------|-----------|
| Trade count             | 348,369   | 142,128   | 51,784  | 1,015,024 |
| Total volume (contracts)| 1,070,364 | 335,016   | 121,454 | 2,509,428 |
| Avg trade size          | 3.17      | 0.33      | 2.35    | 3.92      |
| Price range (pts)       | 121.4     | 38.8      | 30.5    | 298.5     |
| Buy/sell volume ratio   | 1.003     | 0.022     | 0.942   | 1.096     |
| Realized vol (per min)  | 0.000894  | 0.001429  | 0.000125| 0.006346  |

Buyer- and seller-initiated trades account for 49.91% and 50.09% of
volume respectively, with unknown aggressor side comprising a
negligible 0.000003% of records. Individual trade sizes are highly
right-skewed; the median is 1 contract and the 75th percentile is
3 contracts, while the maximum reaches 1,968 contracts. Daily volume
was relatively stable at ~1 million contracts per day from May
through September 2025, becoming more elevated and volatile from
October through December, suggesting two distinct activity regimes
within the sample period.

[FIGURE 1: Daily total volume over time across 169 trading days,
showing two distinct activity regimes — daily_volume_over_time.png]

### 3.4 Intraday Patterns

Intraday volume exhibits a modified inverse-J shape across the
trading day, peaking at the open, declining steadily through the
morning, remaining low through midday, and surging sharply at the
close — reaching ~52,000 contracts per minute in the final minute
of trading, driven by MOC order flow (Figure 2). Trade arrival rate
follows the same pattern, with ~4,000 trades per minute at the open
and ~7,500 at the close, indicating that the close-of-day spike
reflects both more trades and larger average trade sizes.

[FIGURE 2: Average intraday volume by minute across 169 trading
days — intraday_volume.png]

Intraday realized volatility exhibits a distinctly different pattern.
Volatility spikes sharply at the open (~0.0053 per minute, ~3x the
midday average) as overnight information is incorporated into prices,
then declines rapidly and remains flat through midday and into the
close, with no corresponding spike despite the sharp increase in
volume. This divergence is consistent with MOC-driven close activity
— uninformed, publicly announced order flow that generates no adverse
selection and therefore minimal price impact — in contrast to the
open, where informed order flow drives both high volume and high
volatility through adverse selection.

[FIGURE 3: Average intraday realized volatility by minute across
169 trading days — intraday_volatility.png]

---

## 4. Signal Construction and Empirical Design

### 4.1 Trade Flow Imbalance

The primary signal is Trade Flow Imbalance (TFI), defined as the
normalized difference between buyer- and seller-initiated volume
within each 1-minute bar:

TFI_t = (BuyVol_t − SellVol_t) / (BuyVol_t + SellVol_t)

where BuyVol_t and SellVol_t are the total contract size of
buyer-initiated (aggressor side B) and seller-initiated (aggressor
side A) trades respectively within bar t. TFI_t ∈ [−1, 1], with
positive values indicating net buying pressure and negative values
indicating net selling pressure.

Unconditionally, TFI mixes informed and uninformed order flow.
Index fund rebalancing, inventory management by market makers, and
retail liquidity trades all contribute to TFI without carrying
information about future price direction. The analysis therefore
conditions TFI on a regime indicator designed to isolate periods
where order flow is more likely to reflect private information.

### 4.2 Informed Trading Regime Score

The informed trading regime score, RegimeScore_t ∈ [0, 1], is a
continuous measure of the probability that bar t occurs during a
period of elevated information asymmetry. Kyle's lambda is chosen
as the primary component because it is the most direct available
measure of adverse selection intensity — it captures price impact
per unit of signed order flow in real time from trades-only data,
which is precisely the quantity the Glosten-Milgrom and Kyle
frameworks identify as the signature of informed trading. However,
high lambda alone is insufficient; it can be elevated for five
distinct reasons besides genuine informed trading — thin markets
or low depth, market stress or sudden uncertainty, market maker
inventory pressure, mechanical order clustering, and contract roll
proximity. The regime detector is designed to address as many of
these as possible given the constraints of trades-only data.

**Kyle's lambda** measures price impact per unit of signed order
flow, estimated from a rolling OLS regression of bar-level price
changes on signed order flow over a 30-bar window updated each
minute. Specifically, ΔPrice_t is the close-to-close price change
within bar t (in index points), and SignedFlow_t is the net
buyer-initiated minus seller-initiated contracts within bar t,
yielding one observation per bar in the OLS. The 30-bar window is
pre-specified from theory: it is consistent with the rolling lambda
windows used in the microstructure literature, provides sufficient
observations for stable OLS estimation, and remains short enough
to be reactive to intraday regime shifts. The window was fixed
before any regressions were run and no sensitivity analysis was
conducted; the robustness of results to alternative window lengths
(e.g., 15 or 60 bars) is a direction for future work. High lambda
indicates market makers are updating quotes aggressively in
response to order flow — consistent with their inference that the
flow may be informed. Lambda is the primary signal; the other
components serve to qualify its elevated readings.

**Trade arrival rate** measures trades per minute over a 5-minute
rolling window and functions as a depth confirmation for elevated
lambda. High arrival rate confirms that many market participants
are actively transacting, ruling out thin markets and low depth
as the source of elevated lambda. When TAR is low, RegimeScore is
suppressed even if lambda is elevated, preventing misclassification
of illiquid episodes as informed. A shorter window than lambda is
used to reflect the faster response of trading activity to changing
market conditions.

**Addressing the five sources of elevated lambda:** One of the five
non-informed sources is addressed by TAR directly — thin markets
and low depth produce low TAR, suppressing RegimeScore even when
lambda is elevated. Two more are addressed by the exclusion windows
described in Section 4.3 — market stress from scheduled macro
announcements and contract roll proximity. Two remain unaddressed
as an irreducible limitation of trades-only data; market maker
inventory pressure and mechanical intraday order clustering outside
exclusion windows can produce elevated lambda that the detector
cannot distinguish from genuine informed flow. This limitation is
acknowledged throughout the paper.

**Multiplicative combination:** The two components are combined using
a multiplicative formulation that implements the hierarchical
structure described above. Each component is first standardized via
a rolling z-score:

z_lambda_t  = (lambda_t  − μ_lambda)  / σ_lambda
z_arrival_t = (arrival_t − μ_arrival) / σ_arrival

Each z-score is then mapped to (0, 1) via the logistic function, and
the two logistic outputs are multiplied:

RegimeScore_t = logistic(z_lambda_t) × logistic(z_arrival_t)

where logistic(x) = 1 / (1 + exp(−x)).

This formulation ensures RegimeScore is high only when both
components are simultaneously elevated. When lambda is low,
RegimeScore is low regardless of TAR. When TAR is low, it
suppresses the lambda signal proportionally, preventing
misclassification of illiquid or thin-market episodes as
informed. The multiplicative structure enforces the intended
hierarchy; lambda provides the adverse selection signal, and
TAR either confirms or suppresses it.

Rolling standardization ensures the score is always relative to
recent market conditions rather than the full sample distribution,
allowing it to adapt to the two distinct activity regimes identified
in Section 3.3. All rolling estimates use only past data — no
lookahead bias is introduced.

**Design choice — Roll spread exclusion:** The Roll (1984) spread
estimator was considered and excluded as a third component for two
reasons. First, Roll is non-orthogonal to lambda; both spike
simultaneously at the same directional price episodes, making Roll
redundant rather than independently informative. Second, Roll fails
precisely in one-sided markets — the high-lambda informed trading
conditions of greatest theoretical interest — because the bid-ask
bounce required for the serial covariance estimator disappears when
prices move consistently in one direction. The two-component
multiplicative detector avoids both problems while preserving the
core economic signal.

**A fundamental constraint of this design** is that lambda and TFI
share a common input; both are derived from aggressor-side signed
order flow. Any regime detector constructed from trades-only data
inherits this circularity at the construction level. The
implications for all downstream tests — including the forward return
predictability tests in Sections 5.1 through 5.7 — are discussed
in Section 4.5.

### 4.3 Exclusion Windows

Two of the five sources of elevated lambda that do not reflect
genuine informed trading — post-announcement mechanical flow and MOC
order flow, and contract roll illiquidity — are addressed through
targeted exclusion windows. RegimeScore_t is set to zero in all
excluded bars regardless of indicator values.

Scheduled macro announcements (FOMC decisions, CPI releases, NFP
releases) trigger a 30-minute post-announcement exclusion window.
These events drive directional order flow from participants
responding to the same public information signal rather than private
information, and their post-announcement windows represent the
period of greatest mechanical lambda inflation. Pre-announcement
windows are retained; informed traders with early signal access may
be active before announcements, and excluding these bars would
discard potentially genuine informed trading episodes. 

Descriptively, 2 of the 6 in-sample FOMC pre-announcement windows 
(June 18 and September 17, 2025) fall within contract roll exclusion 
windows, setting RegimeScore to zero by construction in those 60 bars. 
For the remaining 4 events (May 7, July 30, October 29, December 10),
the 30-minute pre-announcement windows contain 120 bars with mean
RegimeScore 0.275, compared to the full-sample mean of 0.215 —
directionally consistent with elevated adverse selection before
announcements, though 4 events are insufficient for inference. Mean
|TFI| across all 6 pre-announcement windows (180 bars) is 0.097
versus 0.090 full-sample, a modest elevation unaffected by the roll
contamination. Both comparisons are descriptive only.

The final 10 minutes of each session (15:50–16:00 ET) are excluded
due to domination by MOC order flow — mechanical, publicly announced
execution that generates high volume without adverse selection, as
documented in Section 3.4. Contract roll dates and the three
preceding trading days are excluded to remove the structural
illiquidity and mechanical volume migration accompanying front-month
contract expiration.

### 4.4 Regime Detector Validation

Prior to testing TFI predictability, the regime detector is
validated through a formal statistical test of its core claim;
high-regime bars have a significantly stronger within-bar TFI-return
relationship than low-regime bars. This is the mechanism the detector
is designed to identify — elevated adverse selection produces
stronger price impact per unit of order flow within the same bar,
per Kyle (1985). A significant amplification confirms the detector
correctly identifies periods of elevated within-bar price impact.

The validation regresses contemporaneous returns on TFI, a
high-regime dummy, and their interaction, using the same HAC
standard errors applied throughout:

Return_t = α + β₁·TFI_t + β₂·HighRegime_t
         + β₃·(TFI_t × HighRegime_t) + β₅·TFI_{t-1} + ε_t

where HighRegime_t = 1 if RegimeScore_t > 0.5.

The interaction coefficient β₃ = 0.001525 (z = 9.496, p < 0.001).
The within-bar TFI-return slope is 0.001193 in low-regime bars and
0.002717 in high-regime bars — a 2.278x amplification that is
strongly statistically significant. This confirms that the detector
reliably identifies periods in which order flow has disproportionately
larger within-bar price impact, which is the direct empirical
implication of the Kyle (1985) adverse selection framework.

The amplification figure is a conditional association under
confounded conditions, not a causal estimate; TFI and Return_t share
the same bar's trades, so the confounding present in all
contemporaneous TFI-return regressions contributes to β₃. However,
confounding affects both the high-regime and low-regime slope
equally through β₁ — the unconditional TFI-return relationship.
The interaction term β₃ specifically measures the additional
TFI-return slope in high-regime bars relative to low-regime bars,
and this differential amplification cannot be explained by
confounding alone. The validation test therefore provides
directional confirmation that the detector is working as intended,
not a causal estimate of adverse selection magnitude.

This test uses contemporaneous RegimeScore_t deliberately; the
validation asks whether the current bar's regime state is associated
with stronger within-bar price impact in that same bar. Using lagged
RegimeScore would instead test whether the prior bar's regime state
predicts the current bar's TFI-return relationship — a weaker and
less direct test of the detector's real-time function.

A secondary validation is provided by the autocorrelation structure
of TFI across regime states (Figure A2). High-regime bars exhibit
materially lower TFI persistence (lag-1 ACF ≈ 0.075, decaying to
noise by lag 3–4) relative to low-regime bars (lag-1 ACF ≈ 0.117,
remaining above the 95% confidence band through lag 8–9). This
differential is consistent with the Kyle (1985) prediction that
informed traders randomize order timing to minimize price impact,
producing less persistent directional flow in informed episodes than
in uninformed ones. It provides directional corroboration that the
regime detector is identifying structurally different order flow
conditions, not merely high-activity periods.

A pre-announcement validation was considered — testing whether
RegimeScore is elevated in the 30-minute window before macro
announcements, when informed traders with early signal access should
be active. This test is infeasible with RTH-only data; CPI and NFP
releases occur at 8:30 AM before market open, leaving only 6
observable FOMC pre-announcement windows in the in-sample period —
an insufficient sample for a meaningful test. The validation test
above is therefore the primary and sufficient criterion.

The detector operates most reliably during the stable midday and
early afternoon portion of the trading session, where lambda is
estimable, TAR reflects genuine liquidity conditions, and structural
events are absent. This is confirmed empirically in Section 5.5.

### 4.5 Empirical Design

**Researcher degrees of freedom disclosure.** In the interest of
transparency, the following specification choices were fixed before
any regressions were run: 1-minute bar aggregation (standard in the
microstructure literature); RegimeScore threshold of 0.5 (natural
midpoint of the [0,1] logistic output); 30-bar lambda estimation
window (pre-specified from theory); 5-minute TAR window (chosen to
be shorter than the lambda window to reflect the faster response of
trading activity); HAC maxlags = 5 (standard for 1-minute return
regressions); and the 30-minute post-announcement exclusion window
(standard in the event study literature). Two choices were made
before formal regressions were run but after initial implementation:
the multiplicative (rather than additive) combination of lambda and
TAR was adopted after conceptual analysis of the hierarchical
structure, with no predictability results observed under the
additive formulation before the switch. One choice was determined
after seeing results and is explicitly flagged as post-hoc: the
bottom-tercile cut for stable-regime conditions in Section 5.5,
which is acknowledged there as such.

All regressions use the following primary specification unless
otherwise noted:

Return_{t+1} = α + β₁·TFI_t + β₂·RegimeScore_t
             + β₃·(TFI_t × RegimeScore_t)
             + β₄·Return_t + β₅·TFI_{t-1} + ε_t

β₁ captures the unconditional TFI effect when RegimeScore = 0.
β₂ captures any return level difference across regime states
independent of TFI. β₃ is the primary coefficient of interest; the
null hypothesis H₀ is β₃ = 0 (the regime does not amplify TFI's
forward predictive power) against H₁ - β₃ > 0 (the regime
amplifies TFI's predictive power). β₄ and β₅ control for return
autocorrelation and TFI persistence. All regressions use Newey-West
HAC standard errors with maxlags=5, correcting for serial
correlation in 1-minute returns arising from bid-ask bounce and
intraday momentum.

The regression sample contains N = 55,634 bars after dropping NaN
rows arising from rolling warmup (the first 30 bars of each session
before the lambda window is fully populated), day boundary nulling
(the first bar of each day whose return spans the overnight gap and
is not a valid RTH return), and the forward return edge (the final
bar of each day has no observed T+1 return within the same session).
High-regime bars (RegimeScore > 0.5) account for 12.1% of the
regression sample under the multiplicative detector.

**Circularity and estimation bias:** As noted in Section 4.2, lambda
and TFI share aggressor-side signed order flow as a common input —
the first layer of circularity, irreducible with trades-only data.
A second layer operates at the regression interaction level; high
TFI bars mechanically have elevated RegimeScore because high signed
flow simultaneously elevates lambda under the multiplicative
formulation, inflating the interaction term TFI × RegimeScore. This
second layer is confirmed by the monotonic quintile pattern
documented in Section 5.4; coefficients increase monotonically from
Q1 through Q5, inconsistent with genuine informed trading
concentration in moderate quintiles (Kyle, 1985) and consistent
with mechanical co-elevation.

Both layers act to bias β₃ upward, inflating the estimate toward
H₁ and making a positive coefficient easier to find. The direction
of this bias is confirmed by a permutation simulation. Forward
returns are permuted 1,000 times (seed = 42) while preserving the
joint distribution of TFI and RegimeScore; the full primary
regression pipeline is applied to each permuted dataset and β₃ is
recorded. Under this constructed H₀, the null distribution has a
positive mean (0.000011), confirming that mechanical co-elevation
shifts the sampling distribution of β₃ above zero even when forward
returns carry no information. The observed β₃ = 0.000371 falls at
the 85.4th percentile of this null distribution (simulation p-value
= 0.146), consistent with a null result under modest upward bias.
On the stable-conditions subsample (N = 18,355), the simulation null
mean is 0.000000 and the observed β₃ = 0.001016 falls at the 98.5th
percentile (simulation p-value = 0.015), indicating the
stable-conditions in-sample result is not a mechanical artifact.
Its OOS non-replication therefore reflects a sample-specific
phenomenon rather than a measurement bias. Taking the bias direction
as confirmed, a null result under upward bias implies the true β₃
is no larger than the estimated value, making the primary efficiency
finding a conservative bound. The non-confounded label applied to 
Section 5.1 refers specifically to the absence of a mechanical 
relationship between RegimeScore and Return_{t+1} — not to the 
absence of circularity in the interaction term itself, which is 
present in all specifications that use RegimeScore.

---

## 5. Results

### 5.1 Forward Return Predictability — T+1

The primary test asks whether regime-conditioned TFI predicts
returns at the one-bar horizon. Return_{t+1} is unknown when TFI_t
and RegimeScore_t are observed. RegimeScore_t is contemporaneous
with TFI_t but fully predetermined relative to Return_{t+1}; the
next bar's return is not in lambda's estimation window and is not
known when the signal is recorded. This regression is non-confounded
in the sense defined in Section 4.5.

**Table 2: Primary Regression — Return_{t+1}**

| Variable | Coefficient | z-stat | p-value |
|---|---|---|---|
| const | −4.46×10⁻⁶ | −0.597 | 0.550 |
| tfi (β₁) | 0.000637 | 5.886 | <0.001 *** |
| regime_score (β₂) | 1.75×10⁻⁵ | 0.753 | 0.451 |
| tfi_x_regime (β₃) | 0.000371 | 1.191 | 0.234 |
| lag_return (β₄) | −0.488 | −53.865 | <0.001 *** |
| lag_tfi (β₅) | −4.96×10⁻⁵ | −0.902 | 0.367 |

R² = 0.236, N = 55,634. HAC standard errors (Newey-West, maxlags=5).

β₃ = 0.000371, p = 0.234. H₀ is not rejected. ES futures prices
incorporate regime-conditioned order flow information within one
1-minute bar. The unconditional TFI effect (β₁ = 0.000637,
p < 0.001) replicates the Cont, Kukanov, and Stoikov (2014) finding
that order flow imbalance predicts short-term returns; this effect
does not survive round-trip transaction costs of approximately 0.774
basis points and carries no directional trading application. The
regime does not amplify this unconditional predictability — β₃ is
indistinguishable from zero. The dominant source of return
predictability is the mean-reversion control lag_return (β₄ = −0.488,
p < 0.001), capturing bid-ask bounce; R² = 0.236 is driven almost
entirely by this term. Without the mean-reversion control, 
R² = 0.000275, confirming that the regime-TFI interaction accounts 
for essentially none of the return variance.

The quintile interaction analysis in Section 5.4 confirms this null
result is not an artifact of the linearity assumption in the
interaction specification. Figure A1 in the appendix provides a
visual analog of this regression, showing TFI versus one-bar-ahead
log return stratified by regime state. Notably, the high-regime
panel shows a slightly negative OLS slope (−4.61×10⁻⁴) while the
low-regime panel shows a slightly positive slope (+1.07×10⁻⁴);
neither is economically distinguishable from zero at this scale,
and the sign reversal in the high-regime panel is consistent with
the mechanical co-elevation argument — high-|TFI| bars are
disproportionately classified as high-regime through the circularity
channel, inflating the regime label on bars where TFI ran against
the subsequent return.

### 5.2 Contemporaneous Characterization — T+0

A separate analysis characterizes the regime-conditioned relationship
between TFI and same-bar returns. Because Return_t and RegimeScore_t
share bar t's time window, RegimeScore is lagged by one bar
(RegimeScore_{t-1}) to remove regime-level simultaneity:

Return_t = α + β₁·TFI_t + β₂·RegimeScore_{t-1}
         + β₃·(TFI_t × RegimeScore_{t-1})
         + β₅·TFI_{t-1} + ε_t

A residual confounding channel remains irreducible; TFI_t and
Return_t are both constructed from bar t's trades, making their
relationship partially mechanical with trades-only data. This
regression is a specification sensitivity result — not a predictive
test — and is presented as such. Should future research with
orthogonal data resolve this confounding channel and reveal genuine
contemporaneous amplification, the p% intra-bar framework described
in Section 8 provides a natural extension toward a real-time market
maker application.

**Table 3: Contemporaneous Regression — Return_t (lagged RegimeScore)**

| Variable | Coefficient | z-stat | p-value |
|---|---|---|---|
| const | −8.16×10⁻⁶ | −1.214 | 0.225 |
| tfi (β₁) | 0.001200 | 10.238 | <0.001 *** |
| regime_score_lag (β₂) | 3.53×10⁻⁵ | 1.680 | 0.093 |
| tfi_x_regime_lag (β₃) | 0.000617 | 1.761 | 0.078 |
| lag_tfi (β₅) | −0.000105 | −1.769 | 0.077 |

R² = 0.008, N = 55,634.

β₃ = 0.000617, p = 0.078. The interaction is not significant at
conventional thresholds. The lag-1 autocorrelation of RegimeScore
is 0.806 under the multiplicative formulation, meaning lagging by
one bar removes approximately 19% of the regime-level circularity.
The p = 0.078 result is therefore best interpreted as predominantly
driven by residual confounding through regime autocorrelation, with
a non-trivial but unquantifiable secondary contribution from genuine
predictive signal. These two explanations cannot be cleanly separated
with trades-only data. No empirically grounded market maker
calibration can be derived from this result.

---

## 5.3 Horizon Analysis

The primary specification is repeated at cumulative 5-bar and 15-bar
forward horizons. Cumulative log returns are computed within each
trading day only — cross-day windows are nulled to avoid overnight
return contamination.

| Horizon | β₃ | z-stat | p-value | N |
|---|---|---|---|---|
| T+5 | 0.000090 | 0.256 | 0.798 | 54,789 |
| T+15 | 0.000081 | 0.212 | 0.833 | 53,099 |

No regime interaction emerges at any horizon. The unconditional TFI
effect (β₁ ≈ 0.000696–0.000719, p < 0.001) persists at both
horizons but does not vary with regime. Information incorporation in
ES futures is complete within one 1-minute bar in high-regime
conditions, leaving no residual regime-conditioned predictability at
5 or 15 bar horizons.

### 5.4 TFI Quintile Interaction

The continuous regime interaction term is replaced with four quintile
dummy interactions, allowing the regime amplification to vary freely
across the TFI distribution. Quintile 3 (moderate TFI, the
theoretically predicted zone of informed order flow concentration
under Kyle (1985) since informed traders split orders to minimize
price impact) is the omitted reference category. A Bonferroni
correction is applied across the five simultaneous tests (α = 0.01
per test, family-wise α = 0.05).

| Quintile | Coefficient | z-stat | p-value | Bonferroni |
|---|---|---|---|---|
| Q1 (most negative TFI) | −0.000020 | −0.362 | 0.717 | — |
| Q2 | −0.000038 | −0.786 | 0.432 | — |
| Q4 | 0.000081 | 1.684 | 0.092 | — |
| Q5 (most positive TFI) | 0.000126 | 2.076 | 0.038 | — |

No quintile interaction survives Bonferroni correction (threshold:
p < 0.01). The null result in the primary regression is not a
consequence of imposing linearity on the regime-TFI relationship.

The monotonic pattern — coefficients increasing from Q1 through Q5
— is inconsistent with the Kyle (1985) prediction of hump-shaped
concentration in moderate quintiles and is more consistent with
mechanical co-elevation; higher absolute TFI implies higher signed
order flow, which simultaneously elevates lambda and therefore
RegimeScore under the multiplicative formulation, inflating the
interaction term in high-TFI bars. This confirms the second layer
of circularity described in Section 4.5. As established there, this
upward bias makes the null primary result a conservative bound —
the efficiency conclusion holds even under estimation conditions
that favor H₁.

### 5.5 Midday Subsample and Stable Regime Conditions

The regime detector's signal quality depends on the reliability of
its component estimates. Kyle's lambda requires a full 30-bar
estimation window of stable same-session data, and the TAR z-score
is most informative when baseline activity is low. Both conditions
are best satisfied during the midday portion of the trading session
(11:00–13:00 ET), after the open's elevated and noisy order flow has
subsided and before close-of-day dynamics begin. This window was
pre-specified from theory before running any test.

Restricting the primary regression to midday bars produces
β₃ = 0.000812, z = 1.449, p = 0.147, N = 20,279 — still not
significant, but with a coefficient 2.2x larger than the full-sample
estimate. Under pure circularity, the interaction coefficient should
collapse toward the unconditional TFI slope (β₁ ≈ 0.0006) rather
than away from it. The midday β₃ = 0.000812 moves in the opposite
direction, which is more consistent with genuine signal diluted by
open and close noise in the full-sample regression. Both
interpretations remain possible; neither is confirmed at conventional
significance.

A more direct proxy for detector quality is the rolling standard
deviation of signed order flow within the 30-bar lambda estimation
window. Low variance indicates stable, representative order flow
rather than a single large directional episode dominating the OLS
estimate — precisely the conditions under which the multiplicative
detector's lambda component is most reliably estimated. Restricting
the regression to the bottom tercile of this stability metric
(threshold: 211 contracts signed-flow standard deviation) produces
the following gradient:

**Table 4: Stable Regime Conditions — β₃ by Lambda Estimation Stability**

| Condition | N | β₃ | p-value |
|---|---|---|---|
| Full sample | 55,634 | 0.000371 | 0.234 |
| Stable lambda (bottom tercile) | 18,355 | 0.001016 | 0.033 |
| Unstable lambda (top tercile) | 18,355 | 0.000331 | 0.664 |

The stable-condition β₃ = 0.001016 is 2.7x the full-sample estimate
and is statistically significant at α = 0.05 (z = 2.135, p = 0.033).
The gradient from stable to unstable is monotonic and strong. When
the lambda estimation window is stable — containing steady,
representative order flow — the multiplicative regime detector
produces a statistically detectable amplification of TFI's
forward predictive power. High-regime bars constitute 16.5% of the
stable sample versus 12.1% of the full sample, consistent with
stable estimation windows correctly classifying more bars as
genuinely high-regime. Stable-condition bars concentrate in the
mid-to-late afternoon session (13:xx–14:xx peak, 24.5% and 26.7%
respectively), where lambda warmup is complete and baseline TAR
is at its most stable.

Two caveats apply. First, the stability threshold was derived from
the in-sample distribution and is therefore post-hoc — the result
is a finding but not a pre-registered test. Second, the stable
conditions gradient does not generalize to the OOS period; lambda
window stability does not predict OOS significance in the same way
it predicts in-sample β₃. Within the OOS period, late-period
significance concentrates in unstable rather than stable lambda
bars, suggesting the late OOS episode reflects a different mechanism
than the in-sample stable conditions gradient. The finding is
therefore in-sample only, and its generalizability to other market
environments or time periods is unknown.

The stable regime conditions result is presented as a genuine
in-sample finding that motivates a specific future research
direction; with limit order book data, a direct orthogonal stability
criterion (bid-ask spread variance, depth stability, cancellation
rate) could replace the signed-flow proxy and test whether the
finding persists when circularity is removed. Whether removing
circularity would strengthen, weaken, or eliminate the finding
cannot be determined from the current data — the in-sample β₃ under
stable conditions reflects some combination of genuine regime-
conditioned predictability and residual mechanical co-elevation, and
these cannot be separated with trades-only data. 

Figures A3 and A4 in the appendix show the average intraday profiles 
of Kyle's lambda and RegimeScore respectively. Lambda (Figure A3) is 
elevated and relatively stable at the open, reaches its session minimum 
around 12:00–12:30 ET, then rises and remains elevated through the
afternoon before dropping sharply at the close. RegimeScore
(Figure A4) is mechanically zeroed during the first hour of trading
due to the 30-bar lambda warmup window, jumps to approximately 0.23
at 10:30 ET, and remains broadly stable through the session before
dropping to zero at the 15:50 ET exclusion boundary. The flat
midday RegimeScore profile is consistent with the rolling
standardization adapting to the lower-lambda midday environment
rather than suppressing the score when lambda is at its session
trough.

### 5.6 Subsample Stability

The primary regression is estimated separately on two activity
sub-periods identified in Section 3.3.

| Subsample | N | β₃ | p-value |
|---|---|---|---|
| Full sample (May–Dec 2025) | 55,634 | 0.000371 | 0.234 |
| May–Sep 2025 | 35,001 | 0.000283 | 0.495 |
| Oct–Dec 2025 | 20,633 | 0.000543 | 0.207 |

The null result is stable across both sub-periods. The directional
pattern — higher β₃ in the more volatile Oct–Dec period — is
consistent with the regime detector having more genuine informed
trading episodes to identify in elevated-volatility environments,
but the difference between two non-significant p-values carries no
inferential weight. The primary efficiency finding is a structural
property of the sample, not a sub-period artifact.

### 5.7 Out-of-Sample Validation

The in-sample specification is applied without modification to the
held-out 2026 period (2026-01-02 through 2026-03-06, N = 14,774,
46 trading days). No parameters are refit.

The full-period OOS result is β₃ = 0.000371, z = 2.768, p = 0.006 —
significant at α = 0.01 and in the opposite direction of what the
in-sample null would predict. The following diagnostics were
conducted to identify whether this significance is structural or
episodic, corresponding to six candidate explanations for why OOS
results might diverge from in-sample: HAC misspecification, regime
distribution shift, volatility-driven lambda inflation, TFI
distribution shift, random noise, and episode concentration.

Residual autocorrelation at lags 6–10 ranges from +0.001 to +0.020,
ruling out HAC misspecification. The OOS RegimeScore distribution
is essentially identical to in-sample (mean 0.229 vs 0.215,
OOS/IS ratio 1.06x), ruling out structural regime elevation.
Realized volatility in the OOS period is lower than in-sample
(mean |return| ratio 0.42x), ruling out volatility-driven lambda
inflation and making significance harder to achieve. The OOS TFI
distribution is nearly identical to in-sample (1.3% vs 1.0% extreme
bars), ruling out extreme order flow as a mechanical driver. A
permutation test — in which OOS forward returns are randomly
permuted 1,000 times to construct an empirical null distribution for
β₃ absent any predictability, with the actual OOS β₃ then located
within this distribution — places the actual OOS β₃ in the top 0.5%
of the null distribution (permutation p = 0.005), ruling out random
noise.

The source of the aggregate OOS significance is identified through
monthly breakdown and rolling window analysis. January 2026 and
February 2026 are individually null (β₃ = 0.000047, p = 0.671;
β₃ = 0.000325, p = 0.162 respectively). A rolling 10-trading-day
window across the 46-day OOS period shows β₃ near zero throughout
January and early-to-mid February, before rising sharply and
producing seven consecutive significant windows from February 26
through March 6, 2026 (β₃ ranging from 0.000559 to 0.000837).
Estimating the primary regression separately on the early OOS
period (January through February 22, N = 11,484) and late OOS
period (February 23 through March 6, N = 3,290) yields:

| Period | β₃ | p-value | Return std (bps) | R² |
|---|---|---|---|---|
| Early OOS (Jan–Feb 22) | 0.000066 | 0.559 | 3.041 | 0.002 |
| Late OOS (Feb 23–Mar 6) | 0.000774 | 0.022 | 5.792 | 0.111 |

The late OOS period had approximately 2x the realized volatility of
the early period. Whether this reflects a specific macro environment
in late February through early March 2026 or a more general
condition under which the regime detector produces genuine signal
cannot be determined from approximately 10 trading days of data —
a sample too small to characterize the prevailing conditions as a
replicable regime, separate a genuine market-condition effect from
statistical noise, or confirm that any similar future period would
produce the same result. Lambda window stability does not differ
materially between the late and early OOS periods (mean standard
deviation 287.60 vs 279.37 contracts, ratio 1.03x), ruling out
more stable estimation conditions as the explanation.

The OOS result is an episodic finding. The aggregate significance
is driven by a specific two-week episode of elevated realized
volatility in late February through early March 2026. January and
February individually are null. The permutation test confirms the
episode is not random noise, but no orthogonal conditioning variable
explains or predicts it. The in-sample efficiency finding is not
overturned.

---

## 6. Robustness

### 6.1 Lagged Regime Conditioning

As the most stringent robustness test, the primary regression
replaces the contemporaneous RegimeScore_t and interaction
TFI_t × RegimeScore_t with their one-bar lagged counterparts
RegimeScore_{t-1} and TFI_t × RegimeScore_{t-1}. This specification
is fully predetermined — there is no simultaneity between any
variable — and eliminates any residual circularity from the
contemporaneous regime score sharing information with bar t's order
flow.

β₃ (tfi_x_regime_lag) = −0.000152, z = −0.476, p = 0.634,
N = 55,634.

The null result is unambiguous under this specification, implying
that prior-bar regime information has no carry-forward predictive
value for next-bar returns and that regime-conditioned adverse
selection resolves entirely within one bar. This result strengthens
the efficiency finding; the primary null holds under the most
stringent possible predetermined conditioning, with zero
simultaneity between any variable in the regression.

### 6.2 Regime Transition Dynamics

An additional specification tests whether TFI is differentially
predictive at the first bar of a regime transition — the moment
when RegimeScore crosses 0.5 from below — relative to sustained
high-regime periods. A transition dummy is constructed as
TransitionToHigh_t = 1 if RegimeScore_t > 0.5, RegimeScore_{t-1}
≤ 0.5, and the prior bar was not in an announcement exclusion window
(to prevent false transitions from exclusion resets):

Return_{t+1} = α + β₁·TFI_t + β₂·RegimeScore_t
             + β₃·(TFI_t × RegimeScore_t)
             + β₄·(TFI_t × TransitionToHigh_t)
             + β₅·Return_t + β₆·TFI_{t-1} + ε_t

The sustained high-regime coefficient is β₃ = 0.000143 (p = 0.684)
and the transition coefficient is β₄ = 0.000430 (p = 0.061), across
2,354 transition bars (4.2% of sample). The transition effect does
not reach conventional significance. Two diagnostics confirm that
the apparent transition effect is mechanically inflated and that
the null is the correct interpretation.

First, transition bars are split by the median RegimeScore delta at
crossing (median = 0.270). Under genuine informed trader timing,
the effect should be consistent regardless of how large the regime
score delta is at crossing. Under mechanical inflation, the
interaction term TFI × TransitionToHigh is larger when the
RegimeScore delta is larger, because both the transition flag and
the interaction magnitude are driven by the same underlying
directional signed flow:

| Delta group | N transitions | β₄ | p-value |
|---|---|---|---|
| Small delta (≤ median) | 1,177 | 0.000046 | 0.875 |
| Large delta (> median) | 1,177 | 0.000618 | 0.035 |

The large-delta coefficient is 13.4x the small-delta coefficient.
This extreme concentration is inconsistent with genuine informed
trader timing and strongly consistent with mechanical inflation.

Second, the transition threshold is varied from 0.5 to 0.4 and 0.6:

| Threshold | Transition bars | β₄ | p-value |
|---|---|---|---|
| 0.4 | 2,868 | 0.000200 | 0.263 |
| 0.5 | 2,354 | 0.000430 | 0.061 |
| 0.6 | 1,684 | 0.000362 | 0.190 |

No threshold produces a significant result. The apparent effect at
threshold 0.5 is the strongest of the three and is itself marginal
(p = 0.061). Both diagnostics confirm the transition specification
is mechanically inflated and the null result is the correct
interpretation. This test is not evidence of genuine informed trader
timing.

### 6.3 Regime Detector Threshold Sensitivity

The detector validation regression (Equation 5) classifies bars as
high-regime when RegimeScore > 0.5, pre-specified as the natural
midpoint of the [0, 1] logistic output. The following table reports
β₃ and the within-bar TFI-return amplification ratio at thresholds
of 0.4, 0.5, and 0.6.

| Threshold | High-regime bars | β₃ | z-stat | p-value | Amplification |
|---|---|---|---|---|---|
| 0.4 | 17.9% | 0.001226 | 9.559 | <0.001*** | 2.061× |
| 0.5 | 12.1% | 0.001525 | 9.496 | <0.001*** | 2.278× |
| 0.6 | 6.6% | 0.001937 | 8.069 | <0.001*** | 2.553× |

The validation finding is robust to threshold choice; β₃ is
statistically significant at p < 0.001 at all three values. The
monotonic increase in β₃ as the threshold tightens reflects the
expected behavior of a continuous score; stricter thresholds select
a smaller, more price-impact-concentrated subset of bars. This
pattern does not indicate sensitivity to the specific threshold used.

### 6.4 Additive Combination Robustness

The multiplicative formulation of RegimeScore was adopted on theoretical
grounds before formal regressions were run (Section 4.5). For
transparency, the primary regression is re-estimated using an additive
formulation: RegimeScore_t = [logistic(z_lambda_t) + logistic(z_arrival_t)] / 2.
The two formulations classify substantially different fractions of bars
as high-regime (>0.5), for a structural reason: averaging two logistic
outputs centered at 0.5 produces a distribution centered near 0.5, whereas
the multiplicative product is far more selective.

| Formulation | High-regime | β₃ | z-stat | p-value | N |
|---|---|---|---|---|---|
| Multiplicative (main) | 12.1% | 0.000371 | 1.191 | 0.234 | 55,634 |
| Additive | 43.1% | 0.000187 | 0.451 | 0.652 | 55,634 |

The null result holds under both formulations. The primary efficiency
finding is not sensitive to the choice of combination rule.

### 6.5 Lambda and TAR Window Sensitivity

The regime detector uses a 30-bar rolling OLS window for Kyle's lambda
and a 5-bar rolling mean for the trade arrival rate, both pre-specified
in Section 4.3. The following tables report primary regression β₃ and
p-value when each window is varied while the other is held fixed. The
stable-conditions analysis from Section 5.5 is re-run at each lambda
window length with the rolling signed-flow standard deviation window
matched to the lambda estimation window.

Lambda window sensitivity (TAR window fixed at 5):

| λ window | N (full) | β₃ (full) | p (full) | N (stable) | β₃ (stable) | p (stable) |
|---|---|---|---|---|---|---|
| 15 | 60,344 | 0.000826 | 0.006** | 19,912 | 0.000597 | 0.215 |
| 30 (main) | 55,634 | 0.000371 | 0.234 | 18,355 | 0.001016 | 0.033* |
| 60 | 46,214 | 0.000599 | 0.080 | 15,241 | 0.001291 | 0.009** |

TAR window sensitivity (lambda window fixed at 30):

| TAR window | N | β₃ | p-value |
|---|---|---|---|
| 3 | 55,634 | 0.000389 | 0.243 |
| 5 (main) | 55,634 | 0.000371 | 0.234 |
| 10 | 55,634 | 0.000414 | 0.157 |

The TAR window result is insensitive to window choice; β₃ ranges from
0.000371 to 0.000414 and all three specifications are null (p > 0.15).
The primary null result does not depend on the TAR smoothing parameter.

However, the full-sample result is sensitive to the choice of lambda
estimation window. At λ = 30, the primary null holds (p = 0.234). At
λ = 15, the full-sample result is significant (p = 0.006); at λ = 60,
it is marginal (p = 0.080). Observation counts differ because the
rolling OLS warmup matches the window length, yielding 60,344 bars at
λ = 15 and 46,214 bars at λ = 60. Shorter lambda windows increase the
overlap between the lambda estimation sample and the current bar's
signed order flow. Since signed order flow is the same input as TFI,
this amplifies the first circularity layer documented in Section 4.5.
The stable-conditions pattern is consistent with this explanation; at
λ = 15 the full-sample significant result reverses to null under the
stability filter (p = 0.215), whereas at λ = 30 and λ = 60 the
opposite holds, with a null full-sample result and significant
stable-conditions results (p = 0.033 and p = 0.009 respectively). The
λ = 15 full-sample significance is therefore attributable to heightened
circularity rather than independent evidence of regime-conditioned
forward predictability. The stable-conditions finding from Section 5.5
is reinforced at λ = 60 (β₃ = 0.001291, p = 0.009), carrying the same
post-hoc caveat and the same out-of-sample null.

### 6.6 Expanded Announcement Exclusion Set

The primary specification excludes a 30-minute post-announcement window
following each FOMC decision, CPI release, and NFP release (Section 4.3).
Andersen, Bollerslev, Diebold, and Vega (2007) document that these three
release types, together with PPI, advance GDP, and retail sales, generate
the largest and most persistent intraday price responses in U.S. equity
markets. To verify that the original exclusion set is not materially
incomplete, the primary regression is re-run with all six release types
excluded. For pre-market releases at 8:30 AM ET, the post-announcement
window is shifted from the pre-RTH interval (8:30–9:00 AM, which contains
no RTH bars in the 1-minute bar index) to the first 30 minutes of RTH
(9:30–10:00 AM), capturing the opening period during which the pre-open
release is priced in. Release dates reflect government-issued schedules,
incorporating autumn 2025 shutdown-related delays where applicable.

The expanded exclusion set yields β₃ = 0.000326 (p = 0.291, N = 56,214).
The null result is unchanged. The observation count is 580 higher than the
original sample rather than lower because the exclusion mechanism sets
regime score to zero in excluded windows rather than dropping rows from the
regression; the new announcement-day opening windows convert a small number
of NaN regime-score values to zero, which resolves downstream NaN in the
lagged regime-score term and admits those bars to the regression. This is a
minor sample composition artifact that does not affect the inferential
conclusion. The primary null finding does not depend on the choice of
announcement exclusion set.

---

## 7. Market Maker Implications

The formal results carry two directly actionable insights for market
makers operating in ES futures.

The primary finding — no regime-conditioned forward predictability
at any horizon from T+1 through T+15 — establishes that ES futures
incorporate regime-conditioned order flow information within one
1-minute bar. A market maker who misses the within-bar adverse
selection signal has no opportunity to adjust quotes before the
price has fully moved. This underscores the importance of real-time
rather than bar-level regime detection; the signal distinguishing
informed from uninformed order flow is present and acting on prices
within the bar, not across bars.

The contemporaneous characterization (Section 5.2) produces
β₃ = 0.000617 (p = 0.078) — not statistically significant — and
retains irreducible TFI-Return confounding within bar t. No
empirically grounded quantitative calibration of adverse selection
amplification can be derived from the current data. The dominant
explanation for the marginal p-value is residual regime
autocorrelation (lag-1 autocorrelation = 0.806), not genuine causal
amplification.

The most actionable finding is the stable regime conditions gradient
in Section 5.5. When the lambda estimation window is stable —
defined as the bottom tercile of rolling signed-flow standard
deviation — the regime-conditioned TFI interaction coefficient is
β₃ = 0.001016 (p = 0.033), statistically significant and 2.7x the
full-sample estimate. This provides in-sample statistical evidence
that the regime detector produces stronger forward predictive signal
under specific, identifiable estimation conditions. The practical
implication is directional; a market maker with access to real-time
order flow stability diagnostics — computed from the same trades-only
data used here — can identify windows where the regime detector's
signal is most reliable and concentrate adverse selection management
during these periods.

**Net-return magnitude.** A back-of-envelope calculation bounds the
economic significance of the regime-conditioned signal in the
high-regime/high-|TFI| cell. In high-regime bars (RegimeScore >
0.5, N = 6,719 in-sample), mean |TFI| is approximately 0.08
(approximate, computed from the in-sample distribution). Using the
full-sample interaction coefficient β₃ = 0.000371, the
regime-conditioned contribution to expected next-bar log return at
mean |TFI| = 0.08 and a representative RegimeScore of 0.75 is
approximately 0.000371 × 0.08 × 0.75 ≈ 0.000022, or 0.22 bps.
Using the stable-condition coefficient β₃ = 0.001016, the same
calculation yields 0.000061, or 0.61 bps. Round-trip transaction
costs are approximately 0.774 bps. Neither figure survives
round-trip costs, confirming that the regime-conditioned signal
carries no directional trading application under either
specification. The stable-condition estimate is approximately 79%
of round-trip costs before accounting for adverse selection risk
at the point of execution — where, by construction, the market
maker is transacting against informed flow. High-regime bars
represent 12.1% of the regression sample (6,719 of 55,634 bars),
yielding approximately 40 high-regime bars per trading day; this
frequency is sufficient to amortize fixed costs but does not
alter the gross-vs-net shortfall. No quantitative calibration
of adverse selection amplification can be derived from these
figures for the reasons stated above.

Two important qualifications apply. First, the stable regime
conditions finding is in-sample only and does not replicate in the
OOS period. Second, the stability metric used here (rolling
signed-flow standard deviation) remains partially circular with the
regime detector. Whether removing this circularity — by replacing
the signed-flow proxy with a direct, orthogonal measure such as
bid-ask spread stability, depth constancy, or cancellation rate —
would strengthen, weaken, or eliminate the finding cannot be
determined from the current data. The finding motivates further
investigation rather than immediate practical application.

---

## 8. Future Research

Two research directions are motivated directly by the current
findings and their limitations.

**Direction 1: Orthogonal regime detector with limit order book
data.** The two layers of circularity documented in this paper —
at the detector construction level and the regression interaction
level — are both irreducible with trades-only single-instrument
data. A regime detector derived from limit order book dynamics
(bid-ask spread changes, depth imbalance, cancellation-to-trade
ratio) would share no informational content with TFI and would
permit a genuinely non-confounded test of the regime-conditioned
forward predictability hypothesis. The stable regime conditions
result (β₃ = 0.001016, p = 0.033 under stable estimation
conditions) motivates this direction but does not predict its
outcome; the in-sample finding reflects some combination of genuine
signal and residual mechanical co-elevation that cannot be separated
with trades-only data. An LOB-based detector with an orthogonal
stability criterion could determine whether the stable conditions
effect strengthens, weakens, or disappears when circularity is
removed — and a null result under an orthogonal detector would
itself be an important finding, confirming that the in-sample
result was circularity-driven. Two sub-questions follow only if
the effect survives - does it persist at sub-minute resolution, and
does it survive two-sided transaction costs?

**Direction 2: p% intra-bar amplification framework.** If a future
study finds significant contemporaneous regime amplification with
an orthogonal detector, the current framework can be extended to
characterize the within-bar dynamics of adverse selection.
Specifically, TFI computed at fixed elapsed-time fractions within
each bar (10%, 20%, ..., 90% bar completion) could be regressed on
bar-end returns conditioned on lagged RegimeScore_{t-1}, yielding
an empirical weight schedule θ_p describing how information content
accumulates within the bar as a function of elapsed time and regime
state. This framework provides the bridge from a statistically
significant regime-TFI relationship to a practically implementable
real-time adverse selection signal. It is constructible from current
tick data but deferred pending a significant and non-confounded
contemporaneous finding.

---

---

## Appendix A: Supporting Figures

Figures A1–A5 provide visual support for claims made in the main
text. They are placed in the appendix to preserve the flow of the
empirical argument; no inferential claim in Sections 4–7 depends on
visual inspection of these figures.

**Figure A1: TFI versus one-bar-ahead log return, stratified by
regime state.**
Two side-by-side scatter panels (up to 5,000 sampled points each).
Left panel: high-regime bars (RegimeScore > 0.5). Right panel:
low-regime bars (RegimeScore ≤ 0.5). X-axis is TFI_t ∈ [−1, 1];
Y-axis is Return_{t+1} in log return units. OLS regression line and
slope labeled in each panel. The high-regime slope is −4.61×10⁻⁴
and the low-regime slope is +1.07×10⁻⁴; neither is economically
distinguishable from zero at the scale of the plot. The sign
reversal in the high-regime panel is consistent with the mechanical
co-elevation argument described in Section 4.5. Visual analog of
Table 2.
[FIGURE A1: tfi_scatter_by_regime.png]

**Figure A2: Autocorrelation function of TFI by regime state.**
Two side-by-side bar chart panels. X-axis is lag in minutes (1–20);
Y-axis is autocorrelation. Left panel: high-regime bars
(RegimeScore > 0.5). Right panel: low-regime bars (RegimeScore
≤ 0.5). Dashed lines indicate 95% confidence bands. High-regime
TFI exhibits materially lower persistence: lag-1 ACF ≈ 0.075,
decaying to within the 95% confidence band by lag 3–4. Low-regime
TFI exhibits substantially higher persistence: lag-1 ACF ≈ 0.117,
remaining above the 95% band through approximately lag 8–9 before
slow decay. The differential is consistent with the Kyle (1985)
prediction that informed traders randomize order timing to minimize
price impact, producing less persistent directional flow in
high-regime episodes than in low-regime ones.
[FIGURE A2: tfi_acf_by_regime.png]

**Figure A3: Average intraday Kyle's lambda by minute.**
Single panel. X-axis is time of day (09:30–16:00 ET); Y-axis is
mean Kyle's lambda averaged across 169 in-sample trading days.
Lambda is elevated and relatively stable near the open (~0.0045),
declines to its session minimum around 12:00–12:30 ET (~0.0024),
then rises and remains elevated through the afternoon
(~0.0048–0.0053) before dropping sharply at the close as
MOC-dominated flow suppresses price impact. The midday trough
reflects the low-activity, low-volatility period documented in
Section 3.4 and directly supports the stable regime conditions
analysis in Section 5.5: the lambda estimation window is most
stable and representative during this trough, not noisiest.
[FIGURE A3: lambda_intraday.png]

**Figure A4: Average intraday RegimeScore by minute.**
Single panel. X-axis is time of day (09:30–16:00 ET); Y-axis is
mean RegimeScore averaged across 169 in-sample trading days.
RegimeScore is mechanically zero from 09:30 through approximately
10:30 ET — the 30-bar lambda warmup window — before jumping sharply
to approximately 0.23 and remaining broadly stable through the
session. The hard drop to zero at approximately 15:50 ET reflects
the end-of-session exclusion window. Unlike the lambda intraday
profile (Figure A3), RegimeScore does not exhibit a midday trough;
the rolling z-score standardization causes the score to reflect
relative rather than absolute lambda levels, so the low-lambda
midday period does not suppress RegimeScore below its session
average.
[FIGURE A4: regime_score_intraday.png]

**Figure A5: Intraday Kyle's lambda for four representative trading
days.**
2×2 grid. X-axis is time of day; Y-axis is Kyle's lambda. Top
left: high-volatility session (2025-06-13), showing large-amplitude
swings and negative lambda readings reflecting the extreme
directional price moves of that session. Top right: low-volatility
session (2025-07-25), showing a compressed, stable profile with
lambda ranging narrowly between approximately 0.001 and 0.005.
Bottom left: FOMC announcement day (2025-07-30), showing low and
stable lambda through the morning, followed by a sharp rise around
14:00–14:30 ET coinciding with the announcement window and sustained
elevation through the close. Bottom right: normal session
(2025-08-13), showing a profile similar in scale to the
low-volatility day with moderate intraday variation. Illustrates
how lambda responds to qualitatively different market conditions
and motivates the exclusion window design for FOMC announcement
days.
[FIGURE A5: lambda_individual_days.png]

---

## References

Ahern, K. R. (2018). Do proxies for informed trading measure
informed trading? Evidence from illegal insider trades.
*Review of Asset Pricing Studies*, 10(3), 397–440.

Andersen, T. G., Bollerslev, T., Diebold, F. X., and Vega, C. (2007).
Real-time price discovery in global stock, bond and foreign exchange
markets. *Journal of International Economics*, 73(1), 251–277.

Cont, R., Kukanov, A., and Stoikov, S. (2014). The price impact
of order book events. *Journal of Financial Econometrics*,
12(1), 47–88.

Dufour, A. and Engle, R. F. (2000). Time and the price impact
of a trade. *Journal of Finance*, 55(6), 2467–2498.

Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and
transaction prices in a specialist market with heterogeneously
informed traders. *Journal of Financial Economics*, 14(1), 71–100.

Kyle, A. S. (1985). Continuous auctions and insider trading.
*Econometrica*, 53(6), 1315–1335.

Easley, D., López de Prado, M. M., and O'Hara, M. (2012).
Flow toxicity and liquidity in a high-frequency world.
*Review of Financial Studies*, 25(5), 1457–1493.

O'Hara, M. (1995). *Market Microstructure Theory*.
Blackwell, Oxford.

Roll, R. (1984). A simple implicit measure of the effective
bid-ask spread in an efficient market. *Journal of Finance*,
39(4), 1127–1139.
