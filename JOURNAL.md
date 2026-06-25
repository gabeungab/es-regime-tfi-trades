# Personal Research Journal

*Purpose: Running log of all work sessions (session summary, findings,*
*open questions, AI usage, and next steps for this project).*

---

## 2026-03-24

**Session Summary:**
- Learned more about quant landscape (HFT/MM firms, systematic funds).
- Formalized project direction and blueprint document: microstructure/
  factor research on ES futures trades data with statistical research
  + market participant implications analysis.
- Set up GitHub repo (quant-research-project), cloned locally to
  Desktop, opened in Cursor.
- Purchased ~9 months of ES tick/trade data through Databento
  (using free credit).
- See commits for repo work specifics (repo + markdown files setup).

**Findings:**
None

**Open questions:**
None

**AI Usage:**
- Used Claude for project brainstorming, workflow guidance, and
  generating blueprint document.

**Next step:**
Start Phase 0, Stream A.

---

## 2026-03-25

**Session Summary:**
- Successfully loaded one day of ES futures trades data into a pandas
  DataFrame using the Databento library.
- Wrote load_day(), load_all_days(), resample_to_bars(),
  remove_outliers(), compute_tfi(), compute_returns(), run_ols(),
  and test_autocorrelation() functions in src/data_loader.py.
- Wrote CLAUDE.md to provide stable project context to Claude Code.
- Wrote NOTES.md to store all my notes on learned technical concepts.
  Added sections "TSS, RSS, and ESS", "R-squared", "F-statistic and 
  Prob(F-statistic)", "t-statistic and P>|t|", "Confidence intervals", 
  "Durbin-Watson statistic", "Newey-West standard errors", and
  "Bracket notation vs dot notation".
- Analyzed run_ols() vs. test_autocorrelation() results and made
  logical connections between the two.

**Findings:**
Data cleaning bug (calendar spreads)
- Problem: Calendar spread instruments (ESM5-ESU5, ESH6-ESM6, etc.) 
  were present in the raw data with spread prices in the 40-50 range, 
  which are not real futures prices. 
- Fix: After calendar spread removal (removing any symbols with 
  hyphens), prices ranged from 5596.00 to 7041.25 (consistent with 
  ES futures price history).

TFI computation bug (uint32 overflow)
- Problem: Initial TFI output showed values in the millions instead of 
  [-1, 1] because buy_volume and sell_volume columns were dtype
  uint32 (unsigned 32-bit integer). When sell > buy, the subtraction
  wraps around a huge positive number instead of going negative.
- Fix: After casting both columns to int64 before computing TFI, TFI 
  range correctly now [-1, 1].

OLS + Ljung-Box results + connection:
- OLS showed TFI has no significant predictive power on forward
  returns (p=0.537, R-squared≈0).
- Ljung-Box revealed TFI is strongly autocorrelated at all lags
  (p≈0), meaning consecutive bars are not independent observations.
- Regression residuals are even more strongly autocorrelated
  (lb_stat≈14,500 at all lags), confirming the model left
  enormous systematic pattern unexplained (R-squared=0).
  - If the model had captured all systematic patterns in
  returns, residuals would be pure noise with no autocorrelation. 
  Since residuals are highly autocorrelated, the "unexplained part 
  of returns differing from the model" (return residuals) actually 
  have a pattern to them, meaning our model isn't picking up on 
  that said pattern.
  - Residuals of this model being autocorrelated doesn't mean we 
  actually found correlation between returns and TFI; this just
  confirms there is autocorrelation between residuals themselves.
  This is consistent with OLS p-value = 0.537 (TFI in not a 
  significant predictor of forward returns in this specification).

**Open questions:**
Note to self: get better at using Python imported ibraries and tools.

**AI Usage:**
- Claude helped identify errors during debugging (KeyError:
  'price', calendar spreads causing bad prices, uint32 overflow 
  in TFI, etc).

**Next step:**
Complete final Stream A skill: write plot_overview() function and
produce a clean 2x2 multi-panel matplotlib figure saved to 
/results/overview.png. Then move onto Stream B, C, and phase 1.

---

## 2026-03-26

**Session Summary:**
- Wrote plot_overview() function in src/data_loader.py and generated 
  results/overview.png.
- Wrote compute_daily_stats() function in src/data_loader.py.
  - Found low-trade-count days and filtered them in load_all_days().
  - Found duplicate trades but confirmed they are valid (see 'Findings').
- Wrote plot_phase1() function in src/data_loader.py.
  - Analyzed the five plots to both understand my data more in depth
    and possibly find any research-worthy paths (see 'Findings').
- Learned about market making mechanics, profits, and risks (see 
  NOTES.md) and wrote RESEARCH.md (section 3.1 to 3.4).

**Findings:**
results/overview.png analysis
- TFI vs time quickly scatters around 0. Makes sense.
- Log returns vs time also scatters around 0, with one large spike
  down near market close (just before 16:00 EST). Also makes sense.
- Frequency vs TFI (TFI Distribution) is centered at zero with skinny 
  tails. Makes sense and follows that TFI vs time scatters around 0.
- Frequency vs Log Returns (Log Returns Distribution) is centered
  sharply around 0 with only 2 real outliers (|x|<0.01). Makes sense.
No real findings here; expected with such simple conditions.

compute_daily_stats() output analysis
- Found four days with abnormally low trade counts in 2025 data (see
  data/README.md) and found them to be bank holidays / half trading
  days. Filtered them out of dataset in the load_all_days() function.

Duplicate trades analysis
- From simlpe print statements in test_load.py, 2,856,579 trades were
  found to share timestamps with at least one other trade (duplicates). 
  Investigated and confirmed as legitimate multi-level fills — large 
  orders sweeping multiple price levels simultaneously. All trades 
  retained in sample. 

Plot analysis (intraday close volume/trade count vs. close volatility):
- Trade arrival rate and volume both show a modified U-shape: high at
  open (~4,000 trades/min, ~12,000 contracts/min), declining through
  midday, then a massive spike at close (~7,500 trades/min, ~52,000
  contracts/min). The close spike dominates, making the shape more
  like an inverse-J.
- Intraday volatility shows a different pattern: sharp spike at open
  (~0.0053) then flat and consistent through the rest of the day
  (~0.0015-0.002) including at close — a small normal-J shape with
  no close spike.
- Key difference: high volume and trade count at close does NOT
  produce high volatility — explained by adverse selection dynamics
  around MOC order flow (see NOTES.md - Market Making).
- Research implication: the close is a special regime where the
  volume-volatility relationship breaks down. I will probbably exclude
  the final 1-2 minutes from formal analysis depending on research
  question, or condition on time-of-day explicitly.
  - These findings don't particularly generate any paths to possible
    research ideas, which is expected for simple conditions like these.
    Markets having higher volatility / more informed trading at open
    versus at close is just a known market charactersitic and isn't unique
    to any IV (ex. TFI) and its predictive power on returns.

**Open questions:**
None

**AI Usage:**
- Claude helped identify errors during debugging.
- Claude helped me understand why there was no volatility spike
  at market close while still having a volume / trade count spike.

**Next step:**
Start phase 2 - Research Question Selection and Literature Review.
Organize NOTES.md with logical ordering and clear heading.

---

## 2026-03-27

**Session Summary:**
- Wrote first three main sections of paper/phase2_develpoment.md by
  researching with Google Scholar / SSRN.com (see document for more info).

**Findings:**
- TFI in ES futures is theoretically motivated as an informed
  trading signal but is diluted by uninformed flow unconditionally.
  The research question requires conditioning TFI on an informed
  trading regime to isolate periods where TFI is most predictive.

- VPIN considered and rejected as primary regime indicator due to
  four violated assumptions: informed traders trade uni-directionally,
  uninformed traders arrive symmetrically, trade arrival follows
  stationary independent Poisson processes, and information events
  are one-sided. Daily VPIN retained as robustness check only.

- Kyle's lambda selected as primary regime indicator — measures
  actual price response to order flow rather than inferring informed
  trading from imbalance patterns. Theoretically grounded in Kyle
  (1985) and Glosten-Milgrom (1985).

- Lambda disambiguation problem identified: five alternative
  explanations for high lambda beyond informed trading — thin
  markets, market stress, inventory pressure, mechanical clustering,
  contract roll proximity.

- Three conditions selected to address the most significant
  alternatives: Low Roll spread (tightness), High trade arrival
  rate (depth), Exclusion windows (MOC, announcements, roll dates
  extended to 3 days before roll). Two alternatives remain
  unaddressable from trades-only data (inventory pressure,
  mechanical clustering) — acknowledged as limitations that
  bias against finding significant results. Still need to double 
  check this combination for improvement.

- Roll spread estimator has a known failure mode in strongly
  one-sided markets (bid-ask bounce disappears). Trade arrival
  rate compensates as it remains reliable regardless of flow
  directionality — the two conditions are complementary, not redundant.

- Other metrics, estimators, proxies, etc. were found while researching,
  but were either not possible to research with my data or I thought was
  too involved for this project (leading to a lower-quality final product).

**Open questions:**
Review sections 3.4 through 3.6 and continue research about a better 
combination of conditions (both more encompassing/efficient while not 
introducing unneccessary complexity with diminishing returns) that
deals with the ambiguity of high lambda being and informed trading.

**AI Usage:**
- Claude helped me connect conceptual ideas togther between research papers.

**Next step:**
Review and edit/polish sections 3.4 through 3.6.
- Add to sections 3.2 and 3.3 explicitly why VPIN assumptions could not 
  be handled the same way that Kyle's Lambda ambiguity is being handled.
- Fill out all citations with the correct format.
Then move onto completing the rest of phase2_development.md.

---

## 2026-03-28

**Session Summary:**
- Completed paper/phase2_development.md — all sections finalized
  including research question, methodology preview, and references.
- Finalized continuous RegimeScore construction replacing binary
  dummy as primary regime operationalization.
- Updated PAPER.md to include Section 4 (Signal Construction and
  Empirical Design) and Section 5 placeholder (Market Maker
  Implications).

**Findings:**
Research question finalized:
- Primary: Does TFI predict short-term returns in ES futures,
  and is this relationship stronger when RegimeScore is high —
  consistent with informed trading dominating order flow?
- Secondary: How does TFI predictive power decay across horizons
  (1/5/15-min) in high vs low RegimeScore periods, and what does
  this imply about information incorporation speed?

Regime detector design finalized:
- Continuous RegimeScore ∈ [0,1] constructed from rolling
  z-scores of Kyle's lambda, Roll spread (inverted), and trade
  arrival rate, combined via logistic transformation. Adapts
  to structural changes across sample period.
- Binary regime dummy (was the first planned regime dummy) 
  retained as robustness check only (75th/25th percentile 
  thresholds, also tested at 70/30 and 80/20).
- Daily VPIN retained as alternative regime indicator robustness
  check — different theoretical framework, different failure modes.

Primary regression specification:
- Return_{t+1} = α + β₁·TFI_t + β₂·RegimeScore_t +
                 β₃·(TFI_t × RegimeScore_t) +
                 β₄·Return_t + β₅·TFI_{t-1} + ε_t
- β₃ is the primary test statistic.
- Lagged return (β₄) and lagged TFI (β₅) added as controls
  for autocorrelation structure — ensures β₃ reflects genuine
  regime-conditioned predictability, not persistence.

Additional tests planned:
- Regime detector validation (volatility, announcement,
  price impact persistence) before primary test
- Horizon analysis at 1/5/15-minute forward returns
- Effect size decomposition (fraction of bars at high
  RegimeScore levels)
- Subsample stability across full sample and two activity
  sub-periods (May-Sep, Oct-Dec)
- Transaction cost analysis + annualized Sharpe ratio
- Out-of-sample validation on 2026 held-out period

Robustness checks pre-specified:
- Lambda/Roll window: 15 and 60 minutes
- Arrival rate window: 2 and 10 minutes
- Percentile thresholds for binary dummy: 70/30, 80/20
- Forward return: with and without one-bar skip (bid-ask
  bounce correction)

**Open questions:**
None

**AI Usage:**
- Claude helped me connect literature to research design
  decisions made throughout Phase 2.
- Claude helped me develop continuous RegimeScore construction
  logic and all the robustness checks and their structure.

**Next step:**
Deeply review entire project to make broad connections between 
everything I have done so far (understanding status more clearly).
Then start Phase 3 — Exploratory analysis of TFI and RegimeScore on 
the full in-sample dataset.

---

## 2026-03-29

**Session Summary:**
- Built src/signal_construction.py with five functions: compute_lambda,
  compute_roll_spread, compute_arrival_rate, compute_regime_score,
  compute_exclusion_mask.
- Built src/test_signals.py to sanity check all outputs end-to-end.
- Diagnosed and fixed three bugs in signal_construction.py:
  exclusion mask was incorrectly flagging non-RTH bars (119k instead
  of ~1.7k close bars), Roll spread was contaminated by overnight
  gaps at day boundaries, and compute_regime_score was not aligning
  the exclusion mask index before applying it.
- Investigated lambda NaN rate within RTH hours.

**Findings:**
Signal construction confirmed working — key output stats:
- Kyle's lambda: mean 0.0042, std 0.0088, range [-0.19, 0.14]
- Roll spread: mean 5.59 index points, std 14.05 — values are
  inflated at 1-minute resolution but used only as relative
  ranking via z-score, not absolute spread estimate
- Trade arrival rate: mean 168 trades/min, max 8,487 — consistent
  with Phase 1 intraday patterns
- Exclusion mask: 7,487 bars excluded (2.1% of total) — within
  expected range (~1,690 close + ~1,140 announcement + ~6,240 roll)
- RegimeScore: 55,171 non-NaN bars; 42.8% above 0.50,
  25.9% above 0.75; 13.6% set to zero by exclusion mask

Lambda NaN rate investigation:
- 31.5% of core RTH bars (10:00-15:50) have NaN lambda
- Confirmed entirely from zero-variance signed flow, NOT from
  window warmup — window=15 and window=30 produce identical NaN
  counts, ruling out warmup as the cause
- Zero-variance bars are self-selecting as uninformative: perfect
  balance between buyer and seller flow is itself evidence against
  an informed trading regime
- 55,171 valid RegimeScore bars across 169 days (~326 bars/day,
  ~5.4 hours of coverage) — sufficient for rigorous analysis

Roll spread limitations noted:
- Values inflated (~22 ticks mean vs ~1 tick true spread) due to
  1-minute resolution and large intraday price moves
- Day boundary fix applied — overnight gaps no longer contaminate
  serial covariance estimate
- Used only as relative indicator via z-score; absolute level
  does not affect regime score construction

Regime detector theoretical scope clarified:
- Operates most reliably during stable midday conditions where
  lambda is estimable and Roll is not in one-sided failure
- First 30 minutes of each session excluded by warmup (subsumed
  by zero-variance exclusion as confirmed empirically)
- Inventory pressure and mechanical clustering remain as
  unaddressable limitations biasing against finding results

**Open questions:**
None

**AI Usage:**
- Claude helped diagnose exclusion mask over-exclusion bug
  (RTH filter missing from all three exclusion categories).
- Claude helped diagnose lambda NaN source (zero-variance
  vs warmup) via window=15 vs window=30 comparison.
- Claude helped interpret Roll spread inflated values and
  confirm z-scoring makes absolute level acceptable.

**Next step:**
1. Review Task 1 findings before proceeding — rerun
   test_signals.py and verify all key stats match findings
   above + limitations. Specifically confirm:
   - Exclusion mask total: ~7,487 bars
   - RegimeScore non-NaN bars: ~55,171
   - Lambda NaN rate core RTH: ~31.5%
   - window=15 and window=30 lambda NaN counts are identical
2. Task 2 (Phase 3): Plot each component individually —
   lambda, Roll spread, arrival rate, RegimeScore. For each:
   time series across full sample, intraday average by time
   of day, distribution histogram, behavior on individual
   days (high-vol, low-vol, announcement, normal).

---

## 2026-03-30 (Phase 3 - C)

**Session Summary:**
- Finalized Task 1 of Phase 3 (Roll failure analysis + limitations).
- Completed Task 2: generated 16 exploratory plots (lambda, Roll,
  arrival rate, RegimeScore) saved to results/phase3/.
- Completed Tasks 3 and 5: TFI-by-regime plots and regime detector
  validation diagnostics.
- Reframed primary research question (see findings).
- Updated PAPER.md Sections 4.1, 4.4, 4.5 and phase2_development.md.

**Findings:**

Roll failure analysis (Task 1):
- Roll fails in one-sided flow (informed conditions) but does not
  reverse regime signal — lambda and arrival rate still push
  RegimeScore correctly. Effect is quantitative (lower magnitude),
  not qualitative (wrong direction).
- Contrast: lambda zero-variance NaN is genuinely uninformative
  (no directed flow), so NaN exclusion is justified. Roll failure
  in one-sided markets is more significant because it occurs during
  genuinely informed conditions.

16-plot exploratory analysis (Task 2):

Filtering decisions applied to all plots: distributions filter to
RTH bars only and exclude zeros (warmup/Roll-failure/exclusion-mask
artifacts). Lambda distribution keeps negatives (real signal).
Timeseries filter to trading days only. Individual days constrained
to 09:30-16:00 Eastern.

Lambda:
- Distribution: sharply peaked near zero, slight positive skew,
  tails to ±0.20. Three timeseries spikes at late June (~0.095),
  early September (~0.085), late December (~0.140) — all macro
  events. Baseline ~0.005-0.015.
- Intraday: flat after 30-bar warmup (~0.0042-0.0048), midday dip
  (~0.0025 at 12:00-12:30), afternoon recovery, sharp close drop
  consistent with MOC uninformed flow.
- Individual days: high-vol (±0.06) vs low-vol (0.001-0.005) is
  10-20x difference. FOMC day shows large negative lambda pre-
  announcement (13:00-14:00) — participants reducing directional
  exposure before statement.

Roll Spread:
- Distribution: heavily right-skewed, spike near zero from Roll
  failure in one-sided markets (genuine zero estimates, not NaN).
- Intraday: counterintuitive wave pattern (peaks ~11:30 and
  14:30-15:30) is artifact of directional price moves inflating
  serial covariance — not a genuine liquidity signal. Rolling
  z-score partially corrects.
- Timeseries spikes match lambda exactly (late June ~75, Sep ~85,
  Dec ~90) — correlated, reducing independence as regime indicators.
- Individual days: high-vol Roll 20-63 pts vs 0-3.5 on all other
  days. Low-vol values (0-1.4) plausible for ES true spread.

Trade Arrival Rate:
- Distribution (RTH, >0): right-skewed, peak 300-500 trades/min,
  tail to ~4,000+.
- Intraday: modified inverse-J — open ~1,700, midday ~400-500,
  close spike ~2,000. Open lower than Phase 1 (~4,000) due to
  5-bar rolling mean smoothing.
- Individual days: FOMC step change at exactly 14:00 reaching
  ~4,300 trades/min validates announcement date/time.

RegimeScore:
- Distribution (RTH, >0): clean U-shape with trough 0.4-0.6 —
  logistic transformation pushing toward extremes as intended.
- Intraday: zero 9:30-10:30 (warmup), jumps to ~0.53 at 10:30,
  flat ~0.40-0.60 through core hours, drops to zero at 15:50
  (close exclusion). FOMC exclusion visible as flat zero
  ~13:30-14:30 on 2025-07-30.
- Timeseries: daily mean 0.37-0.58, no structural trend — rolling
  z-score adapting correctly across both activity regimes.
- Individual days: rapid bar-by-bar oscillation between 0 and 1 on
  all days — correct behavior. Detector captures episodic informed
  trading at 1-min level, not sustained day-level regimes.

Cross-component: Roll and lambda spike together at all three macro
events — driven by the same price moves, reducing their independence.
Acknowledged limitation of the composite score.

TFI-by-regime plots (Task 3) — 53,967 aligned bars:
- High-regime: 23,602 (43.7%), Low-regime: 30,365 (56.3%).
- T+1 scatter: high-regime slope = -1.52e-04, low-regime = +1.94e-04.
  Sign reversal — high-regime mean reverts, low-regime momentum.
- TFI distributions nearly identical across regimes — detector
  selects on market state, not TFI magnitude.
- ACF: high-regime decays fast (below CI by lag 7-8), low-regime
  slow (elevated through lag 12) — consistent with faster vs slower
  information incorporation.

Regime detector validation (Task 5):
- Contemporaneous slope: high = 1.80e-03, low = 0.97e-03. 1.87x
  amplification confirms higher order flow informativeness in
  high-regime. Partially tautological — lambda in RegimeScore
  directly measures price impact. Not the primary contribution.
- Realized volatility: high std = 0.000998, low std = 0.002229.
  Paradox explained by exclusion mask — announcement/roll windows
  (highest volatility) zeroed into low-regime by construction.
  Decile analysis confirms detector directionally correct: decile 8
  std = 0.001172 elevated above middle deciles (~0.000900). Decile 0
  std = 0.003508 = exclusion mask zeros, not genuine low-regime.
- Multi-horizon slopes: mean reversion concentrated at lag 1. Lags
  2+ tiny and inconsistent. Information incorporated within 1-2 min.

Research question reframed: the primary finding is the sign reversal
at T+1, not the contemporaneous amplification. Reframe: "does the
TFI-return relationship reverse sign across regimes — momentum in
low-regime, mean reversion in high-regime — consistent with fast
information incorporation creating overshoot at high price-impact
moments?"

**Bugs fixed (plot_signals.py):**
- Non-trading day zeros in timeseries → ts_filter parameter.
- Non-RTH zeros in distributions → dist_filter parameter (lambda:
  none; Roll/arrival/RegimeScore: >0).
- Individual days UTC x-axis → tz_convert + tz_localize(None) +
  set_xlim(09:30, 16:00).
- Intraday tick overlap → min(350, n-41) / min(390, n-1).
- 2025-09-15 and 2025-09-17 flat zero RegimeScore → both in
  September roll exclusion window (ESU5→ESZ5 roll 2025-09-18).
  Replaced with 2025-06-13 (high-vol) and 2025-07-30 (FOMC).

**Open questions:**
- Does lag-1 sign reversal survive formal testing with Newey-West
  SE and full controls in Phase 4?
- Is mean reversion large enough to exceed transaction costs?

---

## 2026-03-30 (Phase 4 - IP)

**Session Summary:**
- Completed Phase 4 formal statistical analysis.
- Wrote src/formal_analysis.py — data preparation, primary
  regression, contemporaneous regression, horizon analysis,
  subsample stability, transaction cost analysis.
- Saved all outputs to results/phase4/.
- Reframed research contribution based on formal results.
- Updated PAPER.md Sections 4.1, 4.5, 4.6, 5 and
  phase2_development.md to reflect final findings.

**Findings:**

Regression sample: N = 53,787 bars after NaN/warmup/boundary
removal. High-regime: 23,601 (43.9%), Low-regime: 30,186 (56.1%).
Date range: 2025-05-01 10:30 to 2025-12-30 15:58. 169 trading days.

Primary regression (T+1):
- β₃ (tfi_x_regime) = 0.0001, z = 0.568, p = 0.570. Not
  significant. Regime does not amplify TFI's predictive power
  for next-bar returns.
- β₁ (tfi) = 0.0007, z = 5.121, p < 0.001. Unconditional TFI
  effect significant — consistent with Cont et al. (2014) — but
  does not strengthen in high-regime conditions.
- lag_return = -0.489, p < 0.001. Dominant mean reversion
  driving R² = 0.236 — microstructure bid-ask bounce, not alpha.

Contemporaneous regression (T+0):
- β₃ (tfi_x_regime) = 0.0015, z = 7.214, p < 0.001. Highly
  significant. At mean high-regime score (0.788), TFI slope =
  0.0007 + 0.0015 × 0.788 = 0.00188 — 2.7x amplification.
- This is a validation finding, not a predictive finding.
  Kyle's lambda is built into RegimeScore and is itself a
  contemporaneous price impact measure — partial circularity
  acknowledged. Serves as confirmation the regime detector
  measures what it claims to measure.

Horizon analysis:
- T+5: β₃ = -0.000027, p = 0.902. Not significant.
- T+15: β₃ = -0.000108, p = 0.659. Not significant.
- Unconditional β₁ remains significant at all horizons
  (~0.0007-0.0008, p < 0.001) but does not vary with regime.
- Conclusion: information incorporation in ES futures is complete
  within one 1-minute bar in high-regime conditions.

Subsample stability:
- May-Sep: β₃ = 0.0001, p = 0.704. Oct-Dec: β₃ = 0.0001,
  p = 0.639. Null result consistent across both activity regimes.
  Unconditional β₁ stable at 0.00064-0.00073 across subsamples.

Transaction cost analysis (contemporaneous spec):
- Mean ES price: 6,460.55. One-way cost: 0.387 bps.
  Round-trip: 0.774 bps.
- At p75 high-regime TFI (0.073): gross = 1.476 bps,
  net round-trip = 0.702 bps. Positive after costs.
- Break-even TFI (round-trip): 0.031 at mean high-regime
  RegimeScore of 0.788. ~75% of high-regime bars clear this.
- T+1 spec does not survive round-trip costs even if significant.

Final research framing:
- Primary hypothesis (T+1 predictability): null result.
  β₃ not significant. ES futures incorporate regime-conditioned
  order flow within one 1-minute bar.
- Secondary characterization (contemporaneous price impact):
  significant but partially circular. Primary value is as
  regime detector validation and market maker input.
- Key contribution: validated real-time regime detector for ES
  futures, quantified adverse selection amplification (2.7x at
  mean high-regime), confirmed information incorporation speed.
- Open questions answered from 3/30: sign reversal at T+1 did
  not survive formal testing with controls. Mean reversion was
  an artifact of unconditional exploratory slopes.

**Open questions:**
- Do robustness checks (binary regime dummy, VPIN, window
  lengths, skip-bar) change any conclusions?
- Out-of-sample validation on 2026 held-out data.

**Next step:**
Next couple days are busy (school wise), but the next couple
full work sessions should be analysis, not new work.
- Review all of Phase 3 and 4 findings and decisions made and
  try to find a new angle of research; null results are fine,
  but push harder to see if you can make any more real findings.
- Also double check all work; there may be inconsistencies / 
  holes (long day of work, focus not 100%).
- Only until fully satisfied / path has been relatively exhausted,
  then move onto Phase 4 out-of-sample testing + robustness checks.
    - Very important to spend time here, because once out-of-sample 
      is used, theres no turning back.

---

## 2026-04-05

**Session Summary:**
- Deep review of Phase 3 and Phase 4 from the ground up. Identified
  fundamental design problems in the regime detector and regression
  setup. Made key decisions on what to keep, drop, reframe, and add.
  Reframed the entire research question and paper structure.
- Key decisions: drop Roll from regime detector (lambda + TAR only),
  change announcement exclusion to post-only, reframe paper around
  regime detector as primary contribution with TFI as validation and
  efficiency test, add p% intra-bar amplification framework as Phase 5
  analysis, plan data continuation section around LOB-based orthogonal
  detector.
- Did not commit to Phase 5. Evaluating five research angles first.

**Findings:**

PROBLEM 1 — Roll spread: more problems than it fixes.
- Roll was included to rule out thin markets as source of high lambda.
  Problems: (a) Roll is non-orthogonal to lambda — both spike at same
  directional price events, reducing composite score independence.
  (b) Roll clips to zero in one-sided markets — precisely the most
  informed conditions — systematically underestimating RegimeScore
  when informed trading is most intense. (c) Roll addresses only the
  spread symptom of thin markets; TAR already addresses participation
  and depth more directly. Roll is redundant with TAR and introduces
  two new problems.
- Decision: drop Roll entirely. New regime detector is two-component:
  lambda + TAR only. Literature support: Roll (1984) is known to
  produce undefined estimates when autocovariance is positive, exactly
  the condition in one-sided informed markets.

PROBLEM 2 — Fundamental confounding in the contemporaneous regression.
- TFI_t and Return_t are both constructed from the same bar t trades.
  Every buyer-initiated trade simultaneously increases TFI_t (adds to
  BuyVol) and increases Return_t (pushes price up). The correlation
  between TFI_t and Return_t is therefore an algebraic consequence of
  their shared data generating process — not a discovered relationship.
  β₃ in the contemporaneous regression inherits this confounding
  completely: it measures both genuine causal amplification AND the
  mechanical signed-flow-to-price relationship. Since both components
  are always positive, the 2.7x almost certainly OVERESTIMATES the
  true causal amplification by an unknown amount.
- For a same-bar regression to be non-confounded, X and Y must not
  share the same underlying events. With trades-only single-instrument
  data this is impossible — all microstructure variables within a bar
  are computed from the same trades. A truly non-confounded same-bar
  test would require a dependent variable from a different data source
  (e.g., options IV, cross-asset price response).
- Decision: contemporaneous β₃ = 0.0015, z = 7.214 is real as a
  number but is a confounded characterization, NOT an empirical finding.
  Its only legitimate use is as a historical calibration coefficient
  for the market maker application (see below), with confounding
  explicitly stated. The T+1 forward regression is the genuine
  non-confounded test. The null result (β₃ = 0.0001, p = 0.570) is
  the primary empirical finding.

PROBLEM 3 — TFI-lambda circularity (separate from confounding above).
- Lambda is derived from rolling OLS of price changes on signed order
  flow. TFI is normalized signed order flow. Both share aggressor-side
  volume as raw input. This means the regime detector partially measures
  the same thing it is supposed to condition on — the regime is not
  independent of TFI. This circularity exists at the detector level
  (not just the regression level as in Problem 2) and cannot be removed
  with lag structures. It can only be fixed by using a regime detector
  derived from a completely different data source (LOB, options, cross-
  asset) that does not share the signed flow input.
- Ahern (2018) identifies absolute order imbalance (= |TFI| × TotalVol)
  and TFI autocorrelation as the most accurate informed trading
  detectors from trades-only data. However these ARE TFI variants —
  using them as the regime detector would create maximum circularity.
  Lambda + TAR is therefore the least circular available regime
  detector given trades-only data, even though it is less accurate
  by Ahern's standard. This is a deliberate accuracy-vs-orthogonality
  tradeoff, not an oversight. Must be stated explicitly in limitations.
- Decision: acknowledge that we are building the most orthogonal regime
  detector possible for trades-only data, not the most accurate one.
  A truly orthogonal detector requires LOB or options data. This project
  establishes the trades-only baseline.

PROBLEM 4 — Announcement exclusion windows were too broad.
- Pre-announcement window (−30 min before event) may contain genuine
  informed leakage from traders with early signal access — exactly
  what the regime detector should identify. Excluding it removes the
  most theoretically informed period from analysis.
- Post-announcement window (+30 min after event) is public information
  response, correct to exclude.
- Decision: change from ±30 min to +30 min post-event only.

PROBLEM 5 — Realized volatility threshold for unscheduled stress.
- Considered as a filter for unscheduled market stress episodes where
  lambda spikes for non-informed reasons. Rejected because: realized
  vol and returns are directly related, making a vol-based exclusion
  ad-hoc and potentially removing genuine high-informed-trading events.
  No clean theoretical justification for any specific threshold.
- Decision: do not implement. Acknowledge unscheduled stress as an
  explicit limitation. Exclusion windows handle only scheduled events.

RESEARCH REFRAME — Regime detector is the primary contribution.

Original framing: "does TFI predict returns more strongly in informed
regimes?" Led to a null T+1 result and a confounded contemporaneous
finding.

New framing: the regime detector itself is the primary contribution.
TFI serves two explicitly separated and carefully framed roles:

(a) Contemporaneous characterization (T+0) — confounded, used as
market maker calibration only:
- β₃ = 0.0015, z = 7.214. The total TFI slope is dynamic and
  continuous: dReturn/dTFI = β₁ + β₃ × RegimeScore. At mean
  high-regime score of 0.788 this equals 0.00188, giving a 2.7x
  amplification ratio vs. the unconditional 0.0007 baseline.
- 2.7x is NOT a constant multiplier. It is the ratio at one specific
  RegimeScore value. Every RegimeScore value produces a different
  amplification. Every unit increase in RegimeScore increases the TFI
  slope by β₃ = 0.0015.
- 2.7x almost certainly OVERESTIMATES the true causal amplification
  because the confounded regression mechanically inflates β₃.
- Market maker application: a market maker does not compute TFI from
  a completed bar. They observe evolving TFI trade-by-trade within an
  open bar in real time. At the start of bar t, RegimeScore_{t-1}
  is already known (fully predetermined from the prior bar). As each
  trade arrives mid-bar, TFI accumulates. The market maker applies
  the dynamic slope formula (0.0007 + 0.0015 × RegimeScore_{t-1})
  to accumulating TFI, with weight growing from 0% at bar open to
  100% at bar close as the partial TFI converges to its final value.
  This is applying a historically estimated (confounded) coefficient
  to live evolving flow as a calibration guide — not predicting the
  future. The confounding must be stated explicitly so the 2.7x is
  understood as approximate calibration, not a validated causal number.
- Using lagged RegimeScore_{t-1} for this application is the correct
  and cleaner version: the regime signal is fully predetermined before
  any of bar t's confounded trades begin. This removes the regime-
  level simultaneity even though the TFI-Return confounding within
  the bar remains.

(b) Forward predictability test (T+1) — non-confounded, primary result:
- β₃ = 0.0001, p = 0.570. Null result. Genuine efficiency finding.
- Return_{t+1} is not in lambda's estimation window and is unknown
  when TFI_t is observed. No confounding.
- Consistent across May-Sep (p = 0.704), Oct-Dec (p = 0.639), T+5
  (p = 0.902), T+15 (p = 0.659).
- Interpretation: ES futures prices incorporate regime-conditioned
  order flow within one 1-minute bar. There is no residual forward
  predictability at any horizon tested under any conditioning structure.
- Market making implication of the null: regime-conditioned adverse
  selection is fully realized within the current bar. A market maker
  who misses the within-bar signal has no second chance — there is no
  carry-forward of regime-conditioned adverse selection to the next
  bar. This makes real-time intra-bar response (p% framework) the
  only actionable strategy.

THINGS OMITTED COMPLETELY:
- Sign reversal narrative: did not survive formal testing with
  lag_return control (β₄ = -0.489). Artifact of unconditional
  exploratory slopes.
- TFI ACF finding by regime: partially circular. Not independently
  defensible. Dropped.
- 1.87x amplification number: replaced by 2.7x from formal regression.
- Roll spread as regime component: retired with full justification.

NEW ADDITION LOCKED IN — p% intra-bar amplification framework:
- Constructible from current tick data (nanosecond timestamps already
  available). Not a future research item — Phase 5 analysis.
- Method: for each 1-minute bar, compute TFI at 10%, 20%... 90% of
  elapsed bar time. Run regression of bar-end Return_t on PartialTFI_p
  conditioned on RegimeScore_{t-1} at each interval p. The coefficient
  θ_p as a function of p gives the empirically derived weight schedule —
  how much the partial TFI matters per unit of accumulating flow at
  each stage of bar completion, amplified by the lagged regime score.
- Nuance: early-bar portion of the weight curve is most reliable
  (partial TFI shares fewest trades with remaining return). Late-bar
  portion is most confounded. State this explicitly.
- This is the quantitative market making framework — empirically
  derived, not arbitrarily assumed.

KEY NUMBERS (updated after Roll removal and post-only exclusion rerun):
- T+1: β₃ = 0.0001, p = 0.570 — null, genuine, primary finding.
- Contemporaneous: β₃ = 0.0015, z = 7.214 — confounded calibration,
  2.7x at RegimeScore = 0.788, likely overestimate of true effect.
- Unconditional TFI: β₁ = 0.0007, p < 0.001 — replication of Cont
  et al. (2014) in ES futures.
- Horizon: null at T+5 (p = 0.902) and T+15 (p = 0.659).
- Subsample: null in May-Sep (p = 0.704) and Oct-Dec (p = 0.639).

DATA CONTINUATION SECTION (future research beyond current project):
- The single most important extension: replace the regime detector with
  one derived from LOB data (bid-ask dynamics, cancellation rates,
  depth changes) — none of which share the signed-flow input of TFI.
  This makes the regime detector and TFI fully orthogonal, removing
  the circularity that is impossible to escape with trades-only data.
  With this orthogonal detector, paired with forward return testing,
  there are no confounding or circularity problems remaining.
- Two sub-questions with the orthogonal LOB detector:
  (A) Does regime-conditioned TFI predict forward returns at 1-minute
  resolution? If yes, our null result was a detection precision problem,
  not a market efficiency result.
  (B) Does regime-conditioned TFI predict forward returns at sub-minute
  resolution (10-second, 30-second bars)? Tests whether the efficiency
  finding generalizes across bar sizes.
- Reference Ahern (2018) as the accuracy benchmark for trades-only
  regime detection; explain that LOB-based detection supersedes it
  by enabling orthogonality with TFI that trades-only data cannot
  provide regardless of which trades-based measure is chosen.

**Open questions:**

Five angles to evaluate before Phase 5. Priority order:
- Lagged regime conditioning (Angle 4): use RegimeScore_{t-1} to
  condition TFI_t on Return_{t+1}. Removes all simultaneity. Almost
  certainly null, but strengthens efficiency story across all
  conditioning structures. Run first, run quickly.
- Time-of-day interaction (Angle 3): test whether T+1 null masks a
  significant midday-window (11:00-13:00) interaction where regime
  detector operates most cleanly. Best shot at a non-null result.
- Asymmetric TFI quintile interaction (Angle 1): test nonlinearity
  by regime in TFI quintile dummies rather than linear continuous
  interaction.
- Regime transition dynamics (Angle 2): test whether first bar after
  low-to-high regime transition has different TFI-return dynamics.
- Volume-conditioned TFI (Angle 5): absolute order imbalance instead
  of normalized TFI, conditioned on typical volume for time-of-day.

For each angle: assess new circularity introduced, constructibility
with current data, theoretical motivation, and whether null vs.
non-null changes the paper's primary framing.

**Next step:**
1. Implement Roll removal and post-only exclusion in
   signal_construction.py. Rerun formal_analysis.py. Record new β₃.
2. Evaluate five angles in priority order above.
3. Build p% intra-bar framework from tick data (Phase 5 scope).
4. Update PAPER.md, CLAUDE.md, phase2_development.md after rerun.

---

## 2026-04-06

**Session Summary:**
- Literature review on maximum orthogonality of informed trading
  regime detectors from trades-only data. Confirmed lambda + TAR is
  the best achievable detector given our data constraints.
- Finalized decision to include Angles 1, 2, and 3 from the five
  angles discussed yesterday. Dropped Angle 5.
- Angle 4 (lagged RegimeScore) already locked in from prior session.

**Findings:**

REGIME DETECTOR ORTHOGONALITY — Literature Review:

Core principle: orthogonality to TFI means independence from the
DIRECTION of signed order flow. TFI = (BuyVol − SellVol) /
(BuyVol + SellVol). Any measure depending on which side initiated
trades shares informational content with TFI. Measures depending
only on TIMING, SIZE, or TOTAL VOLUME of trades are maximally
orthogonal.

Candidate 1 — Inter-trade duration / ACD residual (Dufour and
Engle, 2000, Journal of Finance):
- Measures the unexpected component of time between trades after
  removing intraday seasonality and autocorrelation via ACD model.
  Shorter-than-expected durations signal informed trading without
  using direction. Completely direction-agnostic.
- Verdict: TAR already captures this concept (trades per minute ≈
  inverse of mean duration). ACD residual is marginally more precise
  but measures the same underlying idea. Implementation complexity
  not justified for marginal orthogonality gain. TAR retained.

Candidate 2 — Trade size distribution (coefficient of variation
of trade sizes within rolling window):
- Size is direction-agnostic. Informed traders may use clustered
  large/medium trades. Theoretically motivated by stealth trading
  literature (Kyle, 1985).
- Verdict: In ES futures (a highly liquid instrument), trade size
  distribution is dominated by retail tick (size 1), extremely
  right-skewed, and unstable at 1-minute resolution. Empirical
  evidence for size-based detection is weaker in futures than
  equities. Rejected.

Candidate 3 — Realized price variability (direction-agnostic):
- Measures magnitude of price movement independent of direction.
  Informed trading causes price to move more per unit time.
- Verdict: Too noisy as a regime detector. Directly related to
  Return_t which appears as a control in the regression — adding
  it to the regime score creates complications. Also closely
  related to lambda. Rejected.

Candidate 4 — VCV (Variance-to-Count Ratio, Hu et al., 2023):
- Ratio of cross-sectional variance of volume to average volume
  count. Direction-agnostic. Positively correlated with all PIN
  measures in the literature.
- Verdict: Designed for daily or multi-day windows. Unstable at
  intraday 30-minute rolling resolution with our bar structure.
  Wrong time scale for our application. Rejected.

Candidate 5 — No-trade probability / Poisson rate (Easley and
O'Hara, 1992):
- Probability of zero trades in a time interval carries information
  about informed trading presence. Direction-agnostic.
- Verdict: ES futures trade arrival is so high that probability of
  zero trades in any reasonable window is effectively zero. This
  measure is designed for illiquid instruments. Inapplicable.
  Rejected.

FINAL VERDICT on orthogonality:
Lambda + TAR is the best practically achievable regime detector
from trades-only data for ES futures, maximizing orthogonality to
TFI direction. No candidate offers a meaningful improvement over
this two-component detector given our data and time-scale
constraints. This validates the data continuation section argument
that LOB data is required for a truly orthogonal regime detector.
This data limitation will be stated explicitly in the paper.

ANGLE DECISIONS — Locked In:

Angle 4 (Lagged RegimeScore): Already locked in from prior session.
Use RegimeScore_{t-1} to condition TFI_t on Return_{t+1}. Fully
predetermined regime — removes all simultaneity. Almost certainly
null, but strengthens efficiency story across all conditioning
structures tested.

Angle 1 — Asymmetric TFI Quintile Interaction:
- Theoretical motivation: the current regression assumes the
  TFI-regime interaction is linear — every unit of TFI produces
  the same incremental return regardless of whether TFI = 0.05
  or TFI = 0.50. Kyle (1985) does not predict this. Informed
  traders disguise orders and do not trade at maximum TFI.
  Extreme TFI values may reflect uninformed herding or mechanical
  flow rather than informed accumulation. The moderate-high TFI
  range (4th quintile, 60th-80th percentile) may be where
  informed trading concentrates. A linear interaction averages
  across all quintiles and misses this concentration.
- Test: replace continuous TFI in the interaction with quintile
  dummy variables. Specification:
    Return_{t+1} = α + Σ_q (β_q · Q_q · RegimeScore) + controls
  where Q_q is an indicator for TFI falling in quintile q.
  Apply Bonferroni correction (p < 0.01 per quintile to maintain
  family-wise α = 0.05). This is pre-registered and theoretically
  motivated.
- Circularity argument for non-null results: circularity predicts
  monotonically increasing quintile effects — lambda is higher
  when |TFI| is higher, so the mechanical relationship should be
  strongest in extreme quintiles. If instead we find a hump-shaped
  pattern (moderate quintiles significant, extreme quintiles not),
  this is inconsistent with a pure circularity explanation and
  more consistent with genuine informed trading concentration in
  the moderate-high flow range. Both interpretations acknowledged.

Angle 2 — Regime Transition Dynamics:
- Theoretical motivation: sustained high-regime bars and the first
  bar after a low-to-high regime transition may have different TFI
  dynamics. If informed traders front-run regime shifts — trading
  most aggressively precisely as the market transitions into an
  informed state, creating the transition itself — then TFI's
  predictive power should concentrate at transition bars rather
  than sustained high-regime bars. This is consistent with Kyle
  (1985) where informed traders optimally time entry.
- Test: compute binary transition indicator:
    TransitionToHigh_t = 1 if RegimeScore_t > 0.5 and
                         RegimeScore_{t-1} ≤ 0.5; else 0
  Add as interaction term:
    Return_{t+1} = α + β₁·TFI + β₂·RegimeScore
                 + β₃·(TFI × RegimeScore)
                 + β₄·(TFI × TransitionToHigh)
                 + controls
  β₄ tests whether TFI is more predictive at transition bars
  than sustained high-regime bars.
- Circularity argument for non-null results: this is the most
  circularity-robust of the three angles. Circularity predicts
  a smooth continuous effect increasing with RegimeScore level —
  it does NOT predict a specific discrete effect at the 0.5
  crossing. A significant β₄ concentrated at transitions while
  β₃ for sustained high-regime remains null is qualitatively
  different from what circularity predicts and is more consistent
  with genuine informed trading dynamics at regime shifts. Both
  interpretations still acknowledged.

Angle 3 — Time-of-Day Masking (midday window 11:00-13:00):
- Theoretical motivation: the regime detector operates most
  accurately during midday for three independent reasons: (a) lambda
  estimation is most stable — the rolling 30-bar window is filled
  entirely with current-session data, with no overnight gap or
  warmup contamination; (b) TAR z-score is most informative —
  midday has lowest intraday baseline activity, so deviations from
  the rolling mean reflect genuine changes in market participation
  rather than time-of-day structure; (c) structural contamination
  is absent — no overnight information incorporation at open, no
  MOC proximity at close. Removing Roll strengthens this case
  because Roll's artificial midday elevation (artifact of directional
  price moves inflating serial covariance) no longer contaminates
  the composite score.
- Test: restrict regression sample to bars with timestamps between
  11:00 and 13:00 ET. Run primary T+1 specification on this
  subsample. Compare β₃ to full-sample result.
- Circularity argument for non-null results: if midday window
  produces significant β₃ while full sample is null, two
  interpretations exist. (A) The regime detector works better in
  midday, revealing genuine signal diluted by open/close noise in
  the full sample — consistent with the theoretical motivation.
  (B) Midday has stronger circularity because lambda is estimated
  most stably and therefore most tightly correlated with TFI.
  Response: interpretation (B) predicts circularity should be
  worst at the open (highest lambda estimation variance, most
  volatile price moves, strongest signed flow), not midday. A
  midday-specific result is therefore more consistent with
  interpretation (A). Both interpretations are acknowledged in
  the paper; neither can be fully ruled out.

Angle 5 — Volume-Conditioned TFI: DROPPED.
- Theoretically valid (absolute imbalance normalized by typical
  volume is more aligned with informed price impact theory) but
  ROI is not comparable to angles 1, 2, and 3. Does not improve
  orthogonality (still direction-dependent). Intraday volume
  normalization introduces look-ahead bias or instability in early
  sample bars. If primary angles produce null results, this angle
  is very unlikely to reverse them. Moved to future research
  section as a potential refinement with LOB data.

**Open questions:**
None

**Next step:**
1. Implement Roll removal and post-only exclusion in
   signal_construction.py. Update formal_analysis.py to include
   all tests in one complete pipeline run:
   - Primary regression (returns at T+1)
   - Contemporaneous characterization (reutnrs at T)
   - Horizon analysis (T+5, T+15)
   - Subsample stability (May-Sep, Oct-Dec)
   - Transaction cost analysis
   - Out-of-sample validation (2026 held-out, first run)
   - Lagged regime conditioning
   - Midday subsample analysis (11:00-13:00)
   - TFI quintile interaction (with Bonferroni correction)
   - Regime transition dynamics
   Record all key updated values.
2. Update PAPER.md, CLAUDE.md, JOURNAL.md after full rerun
   and all test results are known.
3. Build p% intra-bar amplification framework from tick data
   (Phase 5 / Market making implication scope).

Note to self: once the final paper is complete, check if any
major tests/findings we are plannig to include in the final paper
take away from the credibility of the paper (ex. contemporaneous, 
circular, confoundings, etc.). 
- Remove, clearly reword, and do not emphasize tests/findings 
  that are inherently (and honestly presented as) flawed.
- Then rack brain on any additional larger-scope improvements
  to make the project more complete and defensble.

---

## 2026-04-07

**Session Summary:**
- Implemented Roll removal and post-only announcement exclusion in
  signal_construction.py. Verified test_signals.py output correct.
- Rewrote formal_analysis.py to include all Phase 4 tests in one
  complete pipeline: primary, contemporaneous (lagged regime spec),
  horizon, subsample, OOS, lagged regime conditioning, midday
  subsample, TFI quintile interaction, regime transition dynamics,
  transaction cost analysis.
- Added OOS announcement dates for 2026 (FOMC, CPI, NFP).
- Fixed transition dynamics exclusion mask bug — prior excluded
  bars no longer falsely flagged as regime transitions.
- Ran full pipeline. Recorded all results.
- Resolved p% intra-bar framework scope
- Discussed OOS debugging plan and multi-interpretation sections.
- Identified two additional diagnostic tests to run before
  finalizing interpretations.

**Findings:**

Regime detector update (Roll removed, post-only exclusion):
- Exclusion mask: 7,367 bars (was 7,487) — 120 bar reduction
  from removing pre-announcement windows
- RegimeScore non-NaN bars: 57,175 (was 55,171) — increase from
  removing Roll's NaN propagation
- Fraction > 0.5: 42.1% (was 42.8%)

Regression sample: N = 55,634 bars (was 53,787). Increase from
retaining pre-announcement bars and removing Roll NaN propagation.

Primary regression (T+1) — null result confirmed:
- β₃ = 0.000203, z = 0.964, p = 0.335 — null
- β₁ (unconditional TFI) = 0.000628, p < 0.001 — stable
- lag_return = -0.488, p < 0.001 — dominant mean reversion
- R² = 0.236

Contemporaneous characterization — major change from initial run:
- β₃ = 0.000425, z = 1.830, p = 0.067 — NOT significant
- Initial run had β₃ = 0.0015, z = 7.214, p < 0.001
- Collapse is definitive evidence the initial finding was
  substantially circularity-driven. With Roll removed and
  RegimeScore lagged (t-1), the mechanical relationship breaks
  and the result largely disappears. The 2.7x amplification
  calibration is no longer empirically supported. Market maker
  application reframed: contemporaneous amplification is fragile
  to specification and cannot be presented as a calibrated
  quantitative input.

Horizon analysis — null confirmed:
- T+5: β₃ = 0.000012, p = 0.960
- T+15: β₃ = -0.000098, p = 0.709

Subsample stability — null consistent:
- May-Sep: β₃ = 0.000177, p = 0.524
- Oct-Dec: β₃ = 0.000251, p = 0.399

Out-of-sample validation (2026 Jan-Mar) — unexpected significance:
- β₃ = 0.000239, z = 2.908, p = 0.004, N = 14,774
- In-sample null (p = 0.335) but OOS significant (p = 0.004) —
  opposite of expected. Three candidate explanations: (1) episodic
  significance driven by a few high-volatility weeks in an unusual
  macro environment (government shutdown, geopolitical uncertainty,
  Fed leadership transition); (2) sample composition difference —
  OOS period had elevated implied volatility inflating lambda and
  RegimeScore mechanically; (3) false positive from small sample
  (46 days vs 169 in-sample). Cannot determine which explanation
  is correct without diagnostic testing.

Lagged regime conditioning (robustness) — null confirmed:
- β₃ = -0.000089, z = -0.417, p = 0.677
- Cleanest possible test — no simultaneity of any kind
- Strengthens efficiency finding: no carry-forward of regime-
  conditioned adverse selection to next bar

Midday subsample (11:00-13:00) — directionally interesting:
- β₃ = 0.000549, z = 1.398, p = 0.162, N = 20,279
- Coefficient larger than full-sample (0.000203) but not
  significant. Cannot distinguish genuine signal diluted by
  open/close noise from stronger circularity in midday. No
  additional test can resolve this with trades-only data.

TFI quintile interaction — null, monotonic pattern:
- No quintile survives Bonferroni correction (α = 0.01)
- Q5: p = 0.047 (fails corrected threshold)
- Pattern is monotonic Q1→Q5, not hump-shaped — more consistent
  with circularity explanation than genuine informed trading
  concentration in moderate quintiles

Regime transition dynamics — most novel finding:
- Sustained β₃ (tfi_x_regime): 0.000140, p = 0.512 — null
- Transition β (tfi_x_transition): 0.000275, p = 0.018 — marginal
- 4,376 transition bars (7.9% of sample)
- TFI is more predictive at first bar of low-to-high regime
  transition than during sustained high-regime conditions
- Consistent with Kyle (1985) informed trader timing; but
  circularity interpretation not ruled out (large RegimeScore
  delta at transition could mechanically inflate interaction)
- p = 0.018 significant at α = 0.05, not at α = 0.01. Marginal.

Transaction cost: contemporaneous β₃ not significant so
break-even analysis no longer empirically grounded.

**Open questions:**

OOS diagnostic — five tests pending before finalizing interpretation:
1. Day-by-day rolling β₃ across OOS weeks — is significance
   concentrated in specific episodes?
2. OOS vs in-sample RegimeScore distribution comparison — is
   the OOS sample drawn from the same population?
3. OOS vs in-sample realized volatility comparison — does
   elevated OOS volatility mechanically inflate RegimeScore?
4. OOS result by month (January vs February) — which month
   drives the significance?
5. Permutation test (1,000 shuffles of fwd_return) — is OOS
   β₃ in the tail of the null distribution?

Transition dynamics diagnostic — one test pending:
- Bin transition bars by size of RegimeScore delta at transition.
  Under circularity, larger deltas should produce stronger
  transition coefficients. Under genuine signal, effect should
  be consistent regardless of delta magnitude. This is the only
  available differentiating test with trades-only data.

Decide whether Phase 5 should have implementable coding or goes 
directly into PAPER.md as just implications.
- Based on market making implication strength of any non-null findings.

**Next step:**
1. Reanalyze phase4 findings to grasp ideas more clearly and generate
   more questions to test (literally and figuratively) the findings.
2. Run five OOS diagnostic tests and transition dynamics diagnostic. 
3. Finalize Phase 4, and update JOURNAL.md, CLAUDE.md, PAPER.md, 
   phase4_findings.md accordingly (final numbers + interpretations).
4. Reframe Phase 5: p% framework moved to future research.
   Market maker implications outline rewritten about non-null findings.

---

## 2026-04-09

**Session Summary:**
- Deep review of all Phase 4 test results. Worked through
  conceptual interpretations of every test finding, including
  what each result means for the research question, which
  interpretations can be ruled out, and which remain open.
- Identified two critical issues requiring additional diagnostic
  testing before interpretations can be finalized: the OOS
  significance anomaly (p=0.004) and the transition dynamics
  circularity question.
- Wrote OOS Diagnostic Tests block and OOS Lambda Level
  Comparison block in formal_analysis.py. Not yet run.
  All other diagnostic sections (regime score autocorrelation,
  stable regime conditions test, transition delta magnitude,
  transition threshold robustness) planned but not yet written.

**Findings:**

Conceptual interpretations of all Phase 4 results:

Primary regression (T+1, β₃=0.000203, p=0.335):
- Null result is clean and non-confounded. ES futures incorporate
  regime-conditioned TFI within one 1-minute bar. Primary finding.
- No additional concerns or open questions on this result.

Contemporaneous characterization (T+0, β₃=0.000425, p=0.067):
- p=0.067 is a mixture of two explanations: (1) residual TFI-
  Return confounding within bar t, irreducible with trades-only
  data; (2) possible genuine predictive signal from lagged
  RegimeScore on same-bar TFI-return dynamics. Cannot cleanly
  separate without knowing lag-1 autocorrelation of RegimeScore.
  Need to run reg['regime_score'].autocorr(lag=1) — if >0.85,
  residual confounding dominates; if <0.70, genuine signal may
  contribute. Lag-1 autocorr falls between thresholds, so
  interpretation will be mixed regardless.
- Collapse from β₃=0.0015 (initial run, confounded) to 0.000425
  (lagged regime spec, not significant) is itself a finding:
  the initial contemporaneous result was substantially circularity-
  driven. Market maker calibration application no longer supported.

Horizon analysis (T+5 p=0.960, T+15 p=0.709):
- Clean null across all horizons. No open questions.

Subsample stability (May-Sep p=0.524, Oct-Dec p=0.399):
- Both null and consistent. The directional pattern (higher β₃
  and lower p in Oct-Dec) is consistent with the hypothesis that
  the regime detector operates better in more volatile markets,
  but this cannot be confirmed from comparing two non-significant
  p-values. Do not present as a finding — present only as null
  result consistent across both activity regimes.

OOS validation (β₃=0.000239, p=0.004):
- Significance is counterintuitive given in-sample null.
  Candidate explanations: (1) episodic — driven by a specific
  macro episode in the 46-day window rather than structural
  predictability; (2) structural regime difference — OOS market
  conditions better suited for regime detector accuracy;
  (3) HAC misspecification for short unusual period; (4) false
  positive from small sample; (5) elevated volatility mechanically
  inflating lambda and RegimeScore. Cannot determine which without
  diagnostic tests. Five diagnostic tests planned: residual
  autocorrelation beyond lag 5, RegimeScore distribution
  comparison, realized volatility comparison, result by month,
  rolling weekly β₃. OOS Lambda Level Comparison also planned.

Lagged regime conditioning (β₃=-0.000089, p=0.677):
- Cleanest possible test, no simultaneity of any kind. Null
  result strengthens efficiency finding. Prior-bar regime
  information has no carry-forward predictive value.
- p=0.677 vs p=0.335 in primary regression does not mean
  lagged regime is less predictive than contemporaneous — both
  are null and the difference is sampling variation only.

Midday subsample (β₃=0.000549, p=0.162):
- Not significant, but coefficient is 2.7x larger than full
  sample and p is 3x smaller. Directionally consistent with
  regime detector quality hypothesis.
- Not p-hacking: the 11:00-13:00 window was pre-specified from
  theory (lambda warmup complete, TAR most stable, no open/close
  structural contamination) before running the test.
- Key point ruling out circularity as dominant explanation:
  under pure circularity, the interaction coefficient should
  collapse toward the unconditional TFI slope (β₁≈0.0006),
  not away from it. The midday β₃=0.000549 is larger than
  in-sample β₃=0.000203, moving in the wrong direction for
  a pure circularity story. This argument should be included
  in the paper.
- Future research: with LOB data, the time-based proxy could
  be replaced with a direct orthogonal condition (bid-ask
  spread, depth, cancellation rate) to test whether stable
  regime conditions produce genuine predictability.

TFI quintile interaction (no quintile survives Bonferroni):
- Null result confirms the primary null is not a linearity
  artifact. Should be noted as foreshadowing in Section 4.5.
- Monotonic pattern (Q1→Q5 increasing coefficients) is a
  circularity signal: higher |TFI| → higher lambda → higher
  RegimeScore → mechanically larger interaction. Does not
  predict hump-shaped pattern from genuine informed trading
  theory (Kyle 1985: informed traders split orders, moderate
  TFI should concentrate signal).
- Implication: circularity operates at the interaction level
  within the regression itself, not just at the detector
  construction level. Monotonic pattern adds a second layer
  of evidence for circularity beyond the known detector-level
  issue. Does not invalidate project — documents the trades-only
  data limitation precisely.

Regime transition dynamics (sustained p=0.512, transition p=0.018):
- Most novel finding. TFI more predictive at first bar of
  low-to-high regime transition than during sustained high-regime.
- Consistent with Kyle (1985) informed trader timing.
- Two interpretations not yet resolved: (1) genuine informed
  front-running of regime shifts; (2) circularity through large
  RegimeScore delta at transition mechanically inflating
  interaction. Need delta magnitude diagnostic to distinguish.
  Planned test: split transition bars by median delta, compare
  β₄ in small-delta vs large-delta subsets. Under circularity,
  large-delta β >> small-delta β. Under genuine signal, β should
  be similar regardless of delta size.
- Threshold robustness also planned: test 0.4 and 0.6
  thresholds. If result only appears at 0.5, likely artifact.
  If stable across thresholds, more robust.
- p=0.018 marginal (significant at α=0.05, not α=0.01).
  Small-sample concern: 4,376 transition bars is only 7.9%
  of total sample. Roughly 2x more data (350+ trading days)
  needed to confirm at stricter threshold.

Stable regime conditions test (planned, not yet written):
- Motivated by midday subsample result. Instead of time as
  proxy for regime detector quality, use rolling std of signed
  flow in 30-bar lambda window as a direct stability metric.
  Bottom tercile = most stable lambda estimation conditions.
- Prediction: stable-condition bars should show stronger β₃
  or lower p-value than full sample, consistent with detector
  quality hypothesis. Hour distribution of stable bars should
  peak at midday if the stability criterion captures the same
  structural advantage as the time window.

**Open questions:**

1. What is reg['regime_score'].autocorr(lag=1)? Required to
   resolve contemporaneous interpretation.
2. OOS diagnostic results: which explanation(s) are supported?
   Specifically whether significance is episodic (concentrated
   in specific windows) or structural.
3. Is transition p=0.018 driven by large-delta transitions
   (circularity) or flat across delta magnitudes (genuine signal)?
4. Is transition finding stable across thresholds 0.4 and 0.6?
5. Does stable regime conditions test show stronger β₃ in bottom
   tercile of lambda window stability? And does hour distribution
   of stable bars peak at midday?

**Next step:**
1. Write remaining diagnostic sections in formal_analysis.py:
   regime score autocorrelation, stable regime conditions test,
   transition delta magnitude diagnostic, transition threshold
   robustness.
2. Run complete formal_analysis.py pipeline and record all
   diagnostic results.
3. Resolve all five open questions from diagnostic output.
4. Update phase4_findings.md, PAPER.md, CLAUDE.md, JOURNAL.md
   with finalized interpretations.

---

## 2026-04-10

**Session Summary:**
- Wrote remaining diagnostic sections in formal_analysis.py:
  RegimeScore autocorrelation diagnostic, signed flow extraction
  for lambda stability metric, stable regime conditions test,
  transition dynamics delta magnitude diagnostic, and transition
  threshold robustness (0.4 and 0.6).
- Foundings recorded but not interpreted yet (will do tomorrow).

**Findings:**
None

**Open questions:**
- Same five open questions from 2026-04-09 session:
  1. RegimeScore lag-1 autocorrelation — resolves contemporaneous
     interpretation (circularity vs genuine signal).
  2. OOS diagnostic results — which explanation(s) supported for
     p=0.004 OOS significance?
  3. Transition delta magnitude — does transition β vary with
     delta size (circularity) or stay flat (genuine signal)?
  4. Transition threshold robustness — is p=0.018 finding stable
     at thresholds 0.4 and 0.6?
  5. Stable regime conditions — does bottom tercile of lambda
     window stability produce stronger β₃, and does hour
     distribution peak at midday?

**Next step:**
1. Analyze results and resolve all five open questions.
2. Update phase4_findings.md with finalized interpretations.
3. Plan on how to continue (stay in phase 4 or move to phase 5).

---

## 2026-04-11

**Session Summary:**
- Ran full formal_analysis.py pipeline with all diagnostic tests
  and recorded final results and interpreted all diagnostic outputs.
- Updated phase4_findings.md with complete final results and
  interpretations, all pending items resolved.
- Professionalized formal_analysis.py (formatting, code quality,
  documentation, helper functions, output formatting).
- Updated PAPER.md: rewrote Sections 4.2–4.7 and Section 5 to
  reflect the two-component detector, final results, actual tests
  run, reframed market maker implications, and added Section 6.

**Findings:**

All Phase 4 findings locked in. Full record in phase4_findings.md.
Key resolved items from this session:

RegimeScore lag-1 autocorrelation = 0.8427. Falls between the two
interpretation thresholds, with residual confounding dominant.
Contemporaneous p = 0.067 is best explained as primarily driven by
residual circularity through regime autocorrelation, with genuine
signal a secondary and unquantifiable contributor.

OOS significance (p = 0.004) is episodic. Driven entirely by the
final two weeks (Feb 23–Mar 6, 2026), during which realized
volatility was approximately 2x the preceding period. January and
February individually are null. Permutation test confirms the
late-period result is not random noise. No orthogonal conditioning
variable explains or predicts it. OOS return scale is confirmed
as a genuine market difference (OOS mean |return| = 0.42x in-
sample), not a data issue.

Transition dynamics finding (p = 0.018) is substantially explained
by circularity. Delta magnitude diagnostic: large-delta transition
β = 0.000414 (6.6x small-delta β = 0.000063). Threshold robustness:
significant at 0.4 and 0.5, disappears at 0.6. Both diagnostics
consistently support mechanical inflation over genuine informed
trader timing. Retired as primary finding.

Stable regime conditions: β₃ = 0.000629 (p = 0.074) in bottom
tercile of lambda window stability vs β₃ = 0.000239 (p = 0.587)
in top tercile. Gradient monotonic and consistent with detector
quality hypothesis. Stable bars concentrate in afternoon (13-14:xx),
not midday. Does not replicate OOS. Presented as exploratory only.

Late OOS investigation reached defined dead end: late OOS lambda
stability is marginally higher than early OOS (1.029x ratio), not
lower. Late OOS significance concentrates in unstable lambda bars
(p = 0.050), not stable ones (p = 0.365). No connection between
afternoon stable conditions and the late OOS episode. The two
phenomena have different underlying mechanisms and cannot be unified.

Quintile monotonicity confirms second circularity layer at the
regression interaction level. Upward bias in β₃ makes the T+1
null result a conservative bound — efficiency conclusion holds
even under favorable estimation conditions for the alternative
hypothesis.

**Open questions:**
None

**Next step:**
1. Review final Phase 4 findings and interpretations and plan
   plan for any possible improvements.
2. Write remaining sections of PAPER.md (Section 1: Introduction
   2: Literature Review).

---

## 2026-04-12

**Session Summary:**
- Deep review of PAPER.md. Identified and discussed structural
  issues across Sections 4.2–4.7 and 5–6, agreed on a full
  set of edits including section renumbering, rewriting 4.2,
  strengthening 4.4 validation, fixing circularity framing,
  and repositioning 4.6.5 as the strongest positive finding.
- Redesigned regime detector from additive to multiplicative
  formulation. Wrote and inserted code into signal_construction.py
  and formal_analysis.py (detector comparison diagnostic and
  both validation tests). Fixed truncation bug in
  signal_construction.py.
- Ran full Phase 4 pipeline with new multiplicative detector
  and interpreted all new results. Dropped pre-announcement
  validation test 2 as infeasible with RTH-only data.
- Updated PAPER.md: full section renumbering, all new results,
  rewritten 4.2 and 4.4, corrected circularity framing,
  stable regime conditions repositioned as strongest positive
  finding, transition dynamics updated to outright null.
- Updated phase4_findings.md with all new results.

**Findings:**

Multiplicative detector is materially different from additive:
Pearson correlation 0.923, bar-level classification agreement
69.0%, high-regime fraction 12.1% vs 43.1%. Design choice has
real consequences for all downstream results.

Validation test 1 passed strongly: high-regime TFI-return slope
= 0.002717 vs low-regime 0.001193 (2.278x amplification,
z = 9.496, p < 0.001). Detector correctly identifies periods of
elevated within-bar price impact.

Primary T+1 null preserved: β₃ = 0.000371, p = 0.234. Efficiency
finding unchanged under new detector.

Stable regime conditions now significant: β₃ = 0.001016, p = 0.033
(vs p = 0.074 under additive). This is the strongest positive
finding in the project. Gradient monotonic: stable p = 0.033,
full sample p = 0.234, unstable p = 0.664. Post-hoc threshold,
does not replicate OOS.

Transition dynamics outright null under multiplicative detector:
transition β₄ = 0.000430, p = 0.061 (vs p = 0.018 under additive).
2,354 transition bars vs 4,376. Delta magnitude ratio 13.4x
(large/small). No threshold significant. Null is the correct
interpretation with no ambiguity.

OOS result: β₃ = 0.000371, p = 0.006. Still episodic — all 7
significant rolling windows concentrated in Feb 26–Mar 6.
January p = 0.671, February p = 0.162. Late OOS β₃ = 0.000774,
p = 0.022. Early OOS β₃ = 0.000066, p = 0.559. Unchanged
interpretation from additive detector.

Contemporaneous: β₃ = 0.000617, p = 0.078. Not significant.
Lag-1 autocorrelation = 0.8064 under multiplicative formulation.

**Open questions:**
None

**Next step:**
Complete full read-through of Sections 4–8 of PAPER.md and
consider all remaining observations.
- Also make both Market Maker Implications and Future Research
  sections stronger and more in-depth / in-detail.

---

## 2026-04-13

**Session Summary:**
- Completed full read-through and final edits to PAPER.md:
  applied Section 4.2 paragraph fixes, updated references
  section, prepended Sections 1 and 2, added abstract.
- Discussed and agreed on remaining project steps: code polish
  pass, LaTeX conversion with figures, final PDF.
- Stepped back to assess project overall — discussed what is
  genuinely novel vs. logically following, and received honest
  rating of strengths and weaknesses.
- Planned and wrote README.md.
- Planned and wrote Sections 1 and 2 of PAPER.md.
- Wrote abstract (212 words, kept full length to preserve all
  central ideas).

**Findings:**
- No new empirical findings this session. All Phase 4 findings
  finalized in previous sessions.
- Project assessment: methodology and circularity documentation
  are the strongest attributes. Stable regime conditions finding
  (p = 0.033) and multiplicative detector formulation are the
  most original contributions. Primary null result is well-
  powered and conservatively framed. Main weakness is data
  limitation — trades-only circularity constrains all positive
  claims.

**Open questions:**
None

**Next step:**
1. Code polish pass — signal_construction.py and
   formal_analysis.py (deep); plot_signals.py, test_load.py,
   and test_signals.py (light).
2. Update GitHub repository — add description and topic tags.
3. Write final paper in LaTeX with figures, formatted tables,
   and proper math environments and produce final PDF.

---

## 2026-04-27

**Session Summary:**
- Polished all five source files: test_load.py, test_signals.py,
  plot_signals.py, signal_construction.py, data_loader.py,
  formal_analysis.py and reran all three pipeline scripts.
- Polished GitHub repo (rename to  es-futures-microstructure, 
  added description and topic tags, updated README).
- Produced final paper in LaTeX on Overleaf
- Purchased MBO data (2025-01-05 to 2025-12-28, ~60GB, ~$375)
  via Databento Python API for future and continued research.

**Findings:**
None

**Open questions:**
None

**Next step:**
Address weaknesses in current final paper:
- Lambda window sensitivity analysis (15/45/60/90)
- Circulation bias simulation to demonstrate instead of 
  only acknowledge.
Continue analyzing final paper for more major weaknesses

Begin planning Paper 2 — MBO-based orthogonal regime
detector study.

---

## 2026-05-20

**Session Summary:**
- Planned Project 2 in full over the last 5 days. 
  - Analyzed Project 1 paper and resume strengths/weaknesses
  - Decided on a three-way detector comparison (Detector A:
    trades-only baseline; Detector B: hand-crafted LOB; Detector 
    C: ML-based LOB logistic regression) using MBO data. 
  - Designed the cancellation-based depth-erosion label 
    (orthogonal to TFI by construction). 
  - Structured Phase 0 (10 trades-data quality improvements 
    before MBO is opened) and Phases 1A through 5.
  - Finalized project2_plan.md.
- Created and set up project 2 repository / github page.

**Findings:**
None

**Open questions:**
None

**Next step:**
Start Project 2, Phase 0.

---

## 2026-06-04

**Session Summary:**
For the last 3 days (forgot to make entries / add to git):
- Began Project 2, Phase 0. Created final_improvements.py in src/
  with full pipeline structure (P0-1 through P0-7) and
  results/final-improvements/ output directory.
- Resolved environment issues before first run: python3 now points
  to Python 3.14.4 (no packages); must use python3.13. External SSD
  mounts at /Volumes/X9 Pro/, not ~/Desktop/X9 Pro/.
- Completed P0-1 (R² Decomposition) and P0-2 (Threshold Sensitivity).
  Outputs committed to results/final-improvements/. PAPER.md updated
  with P0-1 clarification sentence (Section 5.1) and new Section 6.3
  for P0-2 threshold sensitivity table.

**Findings:**
P0-1 — R² Decomposition:
- R² without lag_return = 0.000275 vs. 0.236 with it. Mean-reversion
  control accounts for 99.9% of explained variance; regime-TFI
  interaction accounts for essentially none.
- Diagnostic byproduct: removing lag_return produces β₃ = −0.001271
  (highly significant, negative) — omitted variable bias artifact,
  confirming lag_return is essential for proper inference on the
  interaction term.

P0-2 — Threshold Sensitivity:
- Detector validation β₃ significant at p < 0.001 at all three
  thresholds: 0.001226 (0.4, 2.061×), 0.001525 (0.5, 2.278×),
  0.001937 (0.6, 2.553×). Validation result is robust to threshold
  choice; monotonic increase is expected behavior of a continuous score.

**Open questions:**
None

**Next step:**
P0-3 — Additive Combination Robustness.

---

## 2026-06-06

**Session Summary:**
- Implemented P0-3 (Additive Combination Robustness) in
  final_improvements.py. Added _rolling_zscore and _logistic as
  module-level helper functions (reused in P0-4).
- Ran analysis and discussed interpretation of additive vs.
  multiplicative β₃ difference.
- Added Section 6.4 to PAPER.md robustness section with
  two-row comparison table.

**Findings:**
P0-3 additive combination result:
- Additive high-regime (>0.5) share: 43.1% vs 12.1% multiplicative.
  Not a substantive finding; structural property of averaging two
  logistic outputs centered at 0.5, which produces a distribution
  centered near 0.5 by construction.
- Additive β₃ = 0.000187, z = 0.451, p = 0.652 — null.
  Multiplicative β₃ = 0.000371, p = 0.234 — null.
- Primary efficiency finding is formulation-independent. The
  smaller additive β₃ is consistent with either reduced circularity
  amplification or signal dilution from a less selective formulation,
  but cannot be confirmed as either because both are null.

**Open questions:**
None

**Next step:**
P0-4 — Lambda and TAR Window Sensitivity.

---

## 2026-06-14

**Session Summary:**
- Implemented P0-4 (Lambda & TAR Window Sensitivity) in 
  final_improvements.py,replacing the stub with the full lambda sweep 
  [15, 30, 60] and TAR sweep [3, 5, 10] loops, including stable-conditions 
  re-runs at each lambda window length.
- Ran the script; sanity check confirmed λ=30 baseline reproduces primary 
  result (β₃=0.000371, p=0.234).
- Added Section 6.5 to PAPER.md in the robustness section.

**Findings:**
- TAR window is a non-issue: β₃ ranges 0.000371–0.000414 across TAR=3,5,10, 
  all null (p > 0.15). Primary null does not depend on TAR smoothing choice.
- Lambda window shows real sensitivity in the full-sample result: λ=15 
  significant (p=0.006), λ=30 null (p=0.234), λ=60 marginal (p=0.080).
- The stable-conditions pattern reverses at λ=15: full-sample significant 
  but stable null (p=0.215), opposite of λ=30 and λ=60 where full is null and 
  stable is significant (p=0.033 and p=0.009). This is consistent with 
  circularity amplification at short windows driving the λ=15 full-sample 
  result, not genuine signal.
- λ=60 stable-conditions result (β₃=0.001291, p=0.009) is the strongest stable 
  finding across all configurations — reinforces the Section 5.5 result but 
  same post-hoc caveat applies.

**Open questions:**
None

**Next step:**
P0-5 — Expanded Announcement Exclusion Set

---

## 2026-06-16

**Session Summary:**
- Implemented P0-5 (Expanded Announcement Exclusion Set). Expanded
  from FOMC/CPI/NFP to all six Andersen et al. (2007) releases:
  added PPI (×5), retail sales (×4), advance GDP (×2), with
  government shutdown delays applied. Added premarket_rth_shift=True
  so pre-market releases exclude the first 30 min of RTH rather than
  the pre-RTH interval. Added Section 6.6 to PAPER.md.
- Implemented P0-6 (Pre-Announcement Window Characterization).
  Computed mean RegimeScore and mean |TFI| for the 6 FOMC
  pre-announcement windows (13:30–13:59 ET) vs. full-sample means.
  Added descriptive footnote to Section 4.3 of PAPER.md.

**Findings:**
P0-5: β₃ = 0.000326, p = 0.291, N = 56,214. Null unchanged. N is
580 higher than the original 55,634 — the expanded set resolves some
NaN regime_score values to zero on announcement-day opening bars,
admitting them to the regression. Minor artifact, no inferential
consequence.

P0-6: 2 of 6 FOMC events (June 18, September 17) fall within
contract roll exclusion windows; RegimeScore = 0 by construction in
those 60 bars. For the 4 non-contaminated events: mean RegimeScore
0.275 vs. full-sample 0.215 — directionally consistent with elevated
adverse selection, though 4 events are insufficient for inference.
Mean |TFI| across all 6 windows: 0.097 vs. 0.090 full-sample
(1.075×). Reported the two comparisons separately in the footnote;
an aggregate would produce a misleading 0.183 figure driven by the
zeroed events.

**Open questions:**
None

**Next step:**
P0-7 — Formal Bias Simulation.

---

## 2026-06-18

**Session Summary:**
- P0-7 (Formal Bias Simulation): implemented permutation loop in
  `final_improvements.py` (1,000 permutations, seed=42, plain OLS);
  ran on full sample and stable-conditions subsample; updated
  PAPER.md Section 4.5 and Abstract with simulation results replacing
  "plausibility argument / deferred to future work" language
- P0-8 (Market Maker Implications Reframing): rewrote stable-conditions
  paragraph in Section 7 to lead with all three caveats (post-hoc
  threshold, no OOS replication, circular stability metric) before any
  directional framing; removed redundant qualifications paragraph
- P0-9 (Net-Return Magnitude Framing): relabeled 0.22 bps full-sample
  calculation as an upper bound on noise-level magnitude, not a signal
  estimate; separated "neither figure survives" into two sentences
  distinguishing the null upper bound from the caveated in-sample estimate
- P0-10 (Efficiency Interpretation Logic Chain): added conditional
  three-step logic chain to both Abstract and Section 5.1 — 2.278×
  within-bar amplification as basis, "if genuine and not purely
  circularity-driven" conditionality, Phase 3 LOB-based detector as resolver

**Findings:**
- P0-7 full sample: null distribution mean = +0.000011, confirming
  upward bias direction; bias is small (~3% of observed β̂₃ = 0.000371);
  observed at 85.4th percentile (sim p = 0.146) — null holds comfortably
- P0-7 stable subsample: null distribution mean = 0.000000; observed
  β̂₃ = 0.001016 at 98.5th percentile (sim p = 0.015) — the
  stable-conditions in-sample result is NOT a mechanical circularity
  artifact; its OOS non-replication is a sample-specificity problem, not
  a measurement bias issue
- Sim p-values (0.146 full, 0.015 stable) are lower than HAC p-values
  (0.234, 0.033) as expected — plain OLS is anti-conservative under
  serial correlation; both tell the same story

**Open questions:**
None

**Next step:**
- Proof-read PAPER.md
- Update final paper pdf

---

## 2026-06-23

**Session Summary:**
- Proofread full PAPER.md; compiled and produced updated paper PDF 
  in Overleaf.
- Implemented all 10 P0 quality improvements into the Overleaf LaTeX 
  source (main.tex and references.bib), adding four new robustness 
  subsections (§6.3–6.6) and updating the abstract.
- Added Figure A6 (permutation null distribution histogram) to the 
  paper appendix by updating final_improvements.py to produce a 
  paper-quality version.

**Findings:**
None

**Open questions:**
None

**Next step:**
Write resume bullets

---

## 2026-06-24

**Session Summary:**
- Finalized Resume bullets for full project (4 2-line bullets).
  - Bullet 1: Tested ACF secondary validation to be not significant; 
    removed figure A2 and ACF paragraph from PAPER.md and main.tex.
    Final bullet cites 2.278× price impact amplification (p<0.001).
  - 

**Findings:**
Paper's Figure A2 ACF values (0.075 vs 0.117) were computed treating 
sparse high-regime subseries as consecutive — invalid. 
- Correct values: 0.1172 vs 0.1205, diff = −0.0033.

**Open questions:**
None

**Next step:**
Project 1 is done! 
Start project 2 research in the fall (hopefully in MATH 4699)

---
