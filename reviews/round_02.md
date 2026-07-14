# AAAI Review — Round 2
Verdict: weak reject. Biggest issue: two different "SMART" configs share the name across tables
(tab:learned Learned prostate=66.0 vs tab:baselines/fm/effacc SMART prostate=78.2) - undisclosed.

## Consistency (checked)
- +11.7 gain and 70.8 SC learned macro-F1: CONSISTENT & traceable (tab:learned Learned SC mean=70.79, None=59.04, Random=57.48). GOOD.
- tab:baselines SMART wins prostate(78.2)/STAD(52.2): correct, but number != tab:learned Learned(66.0)/tab:ladder MoR-tok(76.3). Disclose the config difference.
- M=128 appears as 59.2 (tab:token) vs 70.8 (tab:mbudget) vs 70.79 (baselines SC) - add caption notes on which router/graph/subset.

## Framing (routing vs smoothing)
- Body/confound now foreground smoothing (good), but TITLE still says "Routing", Conclusion credits "routing" for the +11.7 (which confound attributes to smoothing), pdfinfo/keywords say routing. Retitle to SMOOTHING.

## Top fixes
1. Disambiguate the two SMART configs (label SMART_best = learned-graph + multi-modal).
2. Retitle away from "Routing" -> "Smoothing"; fix Conclusion + pdfinfo.
3. tab:learned mean row: split SC-mean / MO-mean, note 3-vs-10 seed columns.
4. Reconcile M=128 numbers via caption notes.
5. "strongest method" on P-NET is 2/3 cohorts (loses BLCA); qualify. "best on nearly every suite" false for Xin (7/8); soften.
