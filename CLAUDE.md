# CLAUDE.md — Project Context

## What this project is
Studying the regime-conditioned price impact of Trade Flow
Imbalance (TFI) in ES futures. The regime detector is the primary
contribution — a two-component multiplicative composite
(Kyle's lambda × trade arrival rate) that identifies periods of
elevated information asymmetry in real time from trades-only data.
TFI serves two explicitly separated roles: (1) a contemporaneous
characterization of adverse selection amplification, framed as a
confounded specification sensitivity result (not an empirical
finding), and (2) a forward return predictability test, which
produced a null result (β₃ = 0.000371, p = 0.234) — the primary
empirical finding, interpreted as an efficiency result about
information incorporation speed in ES futures.

Key design constraint: with trades-only data, a fully orthogonal
regime detector is impossible — any detector derived from aggressor-
side volume shares informational content with TFI. Lambda + TAR
multiplicative is the least circular available detector given this
constraint. Two layers of circularity are documented: (1) detector
construction level — lambda derived from the same signed flow inputs
as TFI; (2) regression interaction level — high TFI bars
mechanically have elevated RegimeScore, confirmed by monotonic
quintile pattern. Both are stated explicitly in the paper.

Final deliverables:
- Clean GitHub repository (es-futures-microstructure) — COMPLETE
- 27-page research paper in LaTeX/PDF — COMPLETE

## Research goals and standards
This project targets quant researcher and trader internship
applications at top firms.

---

## Project phases
- Phase 0: Foundations — Python (Stream A), Statistics (Stream B),
  Microstructure (Stream C). COMPLETE.
- Phase 1: Data familiarization — load full sample, clean,
  descriptive stats. COMPLETE.
- Phase 2: Research question selection and literature review.
  COMPLETE.
- Phase 3: Exploratory analysis of chosen phenomenon. COMPLETE.
- Phase 4: Formal statistical analysis, in/out-of-sample testing.
  COMPLETE.
- Phase 5: Paper writing — all 8 sections, abstract, references.
  COMPLETE.
- Phase 6: Code polish, LaTeX/PDF production, repository cleanup.
  COMPLETE.

## Current status

**ALL PHASES COMPLETE. PHASE 0 IMPROVEMENTS IN PROGRESS.**

Final paper: `paper/es_regime_tfi_trades.pdf` (27 pages).
Repository: es-futures-microstructure on GitHub.
All code polished, all results regenerated, README updated.

Phase 0 quality improvements (project2_plan.md):
- P0-1 R² Decomposition — COMPLETE
- P0-2 Threshold Sensitivity — COMPLETE
- P0-3 Additive Combination Robustness — COMPLETE
- P0-4 Lambda and TAR Window Sensitivity — COMPLETE
- P0-5 Expanded Announcement Exclusion Set — COMPLETE
- P0-6 Pre-Announcement Window Characterization — COMPLETE
- P0-7 Formal Bias Simulation — PENDING

Next project: Paper 2 — MBO-based orthogonal regime detector study.
MBO data purchased: 2025-01-05 to 2025-12-28 (GLBX.MDP3, ~60GB).
Stored at: /Volumes/X9 Pro/raw-market-data/es-futures/mbo/

## Final results summary

Primary regression (T+1): β₃ = 0.000371, p = 0.234 — null.
Primary efficiency finding. β₃ upward-biased by mechanical
co-elevation (quintile monotonicity), making null a conservative
bound.

Contemporaneous (lagged regime spec): β₃ = 0.000617, p = 0.078
— not significant. RegimeScore lag-1 autocorrelation = 0.8064.
Residual confounding dominant. Market maker calibration not
empirically supported.

Horizon analysis: null at T+5 (p = 0.798) and T+15 (p = 0.833).

Subsample stability: null in May–Sep (p = 0.495) and Oct–Dec
(p = 0.207). Structural null.

OOS validation (2026 Jan–Mar, N = 14,774): β₃ = 0.000371,
p = 0.006. Episodic — driven by final two weeks (Feb 23–Mar 6).
January (p = 0.671) and February (p = 0.162) individually null.
Late-period β₃ = 0.000774, p = 0.022. No orthogonal conditioning
variable explains it. In-sample finding stands.

Lagged regime conditioning: β₃ = −0.000152, p = 0.634 — null.
Cleanest possible test.

Midday subsample: β₃ = 0.000812, p = 0.147 — not significant,
2.2x full-sample. Pre-specified from theory.

Stable regime conditions: β₃ = 0.001016, p = 0.033 —
statistically significant. Strongest positive finding. Post-hoc
threshold (211 contracts lambda window std, bottom tercile).
Does not replicate OOS. Motivates LOB-based future research.

TFI quintile interaction: no quintile survives Bonferroni (α=0.01).
T+1 null not a linearity artifact. Monotonic Q1→Q5 confirms second
circularity layer.

Regime transition dynamics: β₄ = 0.000430, p = 0.061 — null.
Delta magnitude ratio 13.4x (large/small). No threshold significant.
Circularity confirmed as explanation.

---

## Project structure
es-futures-microstructure/
    /data        — data documentation only, no raw files
    /src         — clean reusable Python modules
        data_loader.py         — data loading, cleaning, stats, plotting
        final_improvements.py  — Phase 0 quality improvements pipeline
        formal_analysis.py     — Phase 4 formal regression pipeline
        plot_signals.py        — Phase 3 signal component plots
        signal_construction.py — signal and regime score functions
        test_load.py           — Phase 1 data loading checks
        test_signals.py        — signal output sanity checks
    /results
        /phase1              — Phase 1 outputs
            daily_stats.csv
            daily_volume_over_time.png
            intraday_trade_arrival.png
            intraday_volume.png
            intraday_volatility.png
            trade_size_distribution.png
        /phase3              — Phase 3 exploratory plots
            lambda_timeseries.png          lambda_intraday.png
            lambda_distribution.png        lambda_individual_days.png
            arrival_timeseries.png         arrival_intraday.png
            arrival_distribution.png       arrival_individual_days.png
            regime_score_timeseries.png    regime_score_intraday.png
            regime_score_distribution.png  regime_score_individual_days.png
            tfi_scatter_by_regime.png
            tfi_distribution_by_regime.png
            tfi_acf_by_regime.png
        /phase4              — Phase 4 formal analysis outputs
            primary_regression.txt
            contemporaneous_regression.txt
            oos_primary_regression.txt
            lagged_regime_regression.txt
            midday_regression.txt
            quintile_interaction_regression.txt
            transition_dynamics_regression.txt
            horizon_t5_regression.txt
            horizon_t15_regression.txt
            subsample_full_regression.txt
            subsample_may_sep_regression.txt
            subsample_oct_dec_regression.txt
            key_coefficients.csv
            transaction_cost_analysis.txt
        /final-improvements  — Phase 0 quality improvement outputs
            p0_1_primary_no_lag_return.txt
            p0_2_threshold_sensitivity.txt
            p0_key_results.csv
    /paper
        es_regime_tfi_trades.pdf       — final working paper (27 pages)
        PAPER.md                       — working paper source
        phase2_development.md          — Phase 2 research development
        phase4_findings.md             — Phase 4 complete findings record
    README.md    — project overview for GitHub
    JOURNAL.md   — chronological research session log
    NOTES.md     — general technical concept reference
    CLAUDE.md    — this file

## Markdown file purposes and language standards

JOURNAL.md — chronological session log. Contains: session
summary, findings specific to this project and dataset,
open questions, AI usage, next steps. Language is informal
and personal. Findings section captures project-specific
empirical observations and debugging discoveries — not
general concepts.

NOTES.md — general technical concept reference. Contains
concepts learned during the project that apply universally
— statistics, Python syntax, market microstructure mechanics.
Language explains concepts from first principles. This is
where general knowledge lives, not project-specific results.

PAPER.md — formal working paper source. Final version produced
as es_regime_tfi_trades.pdf.

phase4_findings.md — complete internal record of all Phase 4
test results, interpretations, diagnostic outputs, and design
decisions. More detailed than PAPER.md. Source of truth for
paper writing.

## Research design summary

**Signal:** TFI_t = (BuyVol_t − SellVol_t) / (BuyVol_t + SellVol_t)
Computed at 1-minute resolution from aggressor side field.

**Regime score (multiplicative):**
RegimeScore_t = logistic(z_lambda_t) × logistic(z_arrival_t)
Two components: Kyle's lambda (30-bar rolling OLS) and trade
arrival rate (5-bar rolling). Roll spread removed — non-orthogonal
to lambda, fails in one-sided markets, redundant with TAR.
All z-scores rolling, past data only. Set to 0 in exclusion windows.
High-regime (RegimeScore > 0.5): 12.1% of in-sample bars.

**Exclusion windows:**
- Final 10 min of session (MOC uninformed flow)
- +30 min POST-announcement only (FOMC, CPI, NFP) — pre-
  announcement window retained as potentially informed
- Contract roll dates + 3 preceding trading days

**Contemporaneous characterization:**
Return_t = α + β₁·TFI_t + β₂·RegimeScore_{t-1} +
           β₃·(TFI_t × RegimeScore_{t-1}) +
           β₅·TFI_{t-1} + ε_t
Uses lagged RegimeScore_{t-1}. TFI-Return confounding within the
bar remains irreducible. Specification sensitivity result only.
β₃ = 0.000617, p = 0.078 (not significant).

**Primary regression:**
Return_{t+1} = α + β₁·TFI_t + β₂·RegimeScore_t +
               β₃·(TFI_t × RegimeScore_t) +
               β₄·Return_t + β₅·TFI_{t-1} + ε_t
β₃ is the primary test statistic.
Final result: β₃ = 0.000371, p = 0.234 — null efficiency finding.

**Key papers:**
- Kyle (1985) — Econometrica — lambda and informed trading
- Glosten and Milgrom (1985) — JFE — adverse selection framework
- Cont, Kukanov, Stoikov (2014) — JFEC — OFI predicting returns
- Ahern (2018) — Rev. Asset Pricing Studies — informed trading
  proxies; most accurate proxies are circular with TFI
- Dufour and Engle (2000) — JF — inter-trade duration and
  informed trading; theoretical basis for TAR
- O'Hara (1995) — Market Microstructure Theory, Blackwell —
  informed trading characterized jointly by price impact and
  active participation; theoretical basis for multiplicative design
- Roll (1984) — JF — implicit spread estimator (considered,
  excluded — non-orthogonal to lambda, fails in one-sided markets)

## Data
- Source: Databento, GLBX.MDP3, Trades schema
- In-sample: 2025-05-01 to 2025-12-30 (169 trading days)
  Location: /Volumes/X9 Pro/raw-market-data/es-futures/trades/
            GLBX-20250501-20251231/
- Out-of-sample: 2026-01-02 to 2026-03-06 (46 trading days)
  Location: /Volumes/X9 Pro/raw-market-data/es-futures/trades/
            GLBX-20260101-20260309/
- MBO data (Paper 2): 2025-01-05 to 2025-12-28
  Location: /Volumes/X9 Pro/raw-market-data/es-futures/mbo/
- Fields used: ts_event_et (Eastern time), price, size,
  side (B/A/N)
- Fields ignored: rtype, publisher_id, instrument_id, action,
  depth, flags, ts_in_delta, sequence (all constant or irrelevant)
- Symbol rolls: ESM5 (May–Jun) → ESU5 (Jul–Sep) →
  ESZ5 (Oct–Dec) → ESH6 (Jan–Mar 2026)
- Excluded dates: 2025-05-26, 2025-07-04, 2025-09-01,
  2025-11-27 (holidays/half-sessions)

## Tech stack
Python 3.13, pandas, NumPy, scipy, statsmodels, matplotlib,
databento library

---

## Standards for Claude Code to enforce
- Train/test split is time-based: in-sample 2025, out-of-sample
  2026. Never touch out-of-sample data for specification search.
- Methodology must be pre-registered before formal tests. Flag
  if tests run before plan is documented in JOURNAL.md.
- All regressions require Newey-West standard errors (maxlags=5).
  Flag if default OLS standard errors are used.
- Flag if multiple hypothesis tests run without Bonferroni
  correction.
- Transaction cost analysis required before any result claim.
  ES futures tick size = 0.25 index points (~4 basis points).
- RegimeScore must be set to 0 in all exclusion windows before
  any regression — flag if exclusions are not applied.
- All rolling estimates must use only past data. Flag any
  lookahead bias in rolling window construction.
- All results must be reproducible from /src code on raw data.

## Rules for Claude Code in this project
- Point out methodological errors or oversights
- Do not generate analysis, write paper sections, or interpret
  results
- When I show you an error, explain what caused it before giving
  the fix
