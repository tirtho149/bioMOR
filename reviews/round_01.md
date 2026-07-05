# AAAI Review — Round 1

## Scores
- Soundness 2/4, Contribution 2/4, Presentation 2/4, **Overall: weak reject**, Confidence 4/5.

## Top weaknesses (priority order)
1. **Title/abstract/method say "routing"; the paper's own confound factorial (tab:confound) shows the gain is SMOOTHING, not routing.** Retitle/reframe around "learned gene-graph smoothing." Keep "routing" only for the MoR depth router (efficiency knob).
2. Learned smoother is close to standard low-rank/GCN denoising — novelty under-argued; add positioning + an unconstrained rank-16 linear-map ablation.
3. SMART loses to logistic regression on 8/11 datasets — accuracy contribution is negative; make the abstract acknowledge this (body already does).
4. **Internal numeric inconsistencies**: 70.8 (abstract) vs 72.1 (tab:anchor random-init) vs 70.8 (tab:mbudget M=128); "no-prior 59.0" vs None column mean. Every headline number must map to a specific table cell.
5. "+11.7" is vs the ablated self, while the strong external (linear 74.8) beats SMART — leading with intra-model delta overstates.
6. **tab:effacc "cheaper AND more accurate" conflates two independent changes** — FLOPs from MoR (flat accuracy), accuracy from smoothing; no wall-clock. Split the claim.
7. FM comparison (tab:fm) is 3 tiny OOD cohorts + frozen scGPT — too weak to tabulate as headline; move to appendix / scope as sanity check.
8. "Compute-allocation importance score" contribution has NO measured evidence (fig:mor is explicitly illustrative). Add measured per-gene recursion depths vs curated markers, or drop the claim.
9. ~1 page of method + appendix derives the additive-prior router that experiments discard — compress; state it's a negative up front.

## Length (must reach ≤8-page main body; currently ~14 total)
Move to appendix: **tab:anchor, tab:fm, tab:mbudget, tab:token, tab:param** (keep one-line summaries in body).
Keep in body: tab:learned (factorial), tab:confound (mechanism), tab:baselines (honesty), tab:ladder (efficiency).
Tighten: abstract (~200 words), contributions (6→4 bullets), Related Work name-drops, Broader Impact (5 paragraphs → 1), shrink/move fig:mor.

## Questions
- Why "routing" title when routing doesn't help?
- Reconcile 70.8/72.1/70.8.
- Is the smoother distinct from a learned rank-r linear map? (ablation)
- Measured recursion depths vs markers?
- Wall-clock speedup?
- When is SMART's accuracy (not interpretability) the right tool?
