# PaperReview.ai (Stanford ML Group) — SMART — submitted 2026-06-23

> AI-generated review (may contain errors). Archived verbatim for the record.
> Companion: `action_items.md` tracks how each point was resolved.

## Summary
SMART: a transformer for transcriptomic classification making parameter efficiency
an architectural property via (i) a learnable cross-attention "marker router"
selecting M<<N marker genes end-to-end; (ii) representing each sample only by
selected marker tokens (O(N^2)->O(M^2)); (iii) a single block applied recursively
with a Mixture-of-Recursions router for adaptive per-marker depth. Strong on
pan-cancer cohort classification with far fewer transformer-stack params, plus
interpretable signals (marker identity, recursion depth). Ablations on routing,
selection, depth; extra eval on clinical phenotype tasks and single-cell.

## Strengths
- Coherent, biologically motivated integration of marker selection + compression +
  weight-shared adaptive recursion; recursion depth as intrinsic importance is neat.
- Peaked init + temperature-annealed all-gene softmax is a thoughtful design.
- Multiple ablations; breadth across clinical + single-cell; compute/param accounting.
- Progressive, clear method exposition; explicit about limitations; reproducible.

## Weaknesses
- On the easier cohort task, recursion/refinement don't improve accuracy; gains are
  efficiency + interpretability, not predictive performance.
- Savings reported for transformer stack only; total size (incl. embeddings) and
  end-to-end wall-clock/throughput not quantified.
- Under-specified: L_comp references s_i (undefined); FLOPs methodology not detailed.
- Key inconsistencies: (i) Table 2 lists PRAD instead of THCA; (ii) text says
  expert-choice within ~1 F1 of fixed-depth but Table 4 shows expert >= fixed;
  (iii) text says router best in selection study but Table 5 shows Concrete higher.
- Cohort task near-saturated; missing stronger baselines (GexBERT, efficient-attention
  / same-budget standard transformer) and multi-seed on the primary task.
- Single-cell Pancreas uses genomap pixels as "genes" -> invalidates marker prior;
  off-distribution; frame more cautiously.
- Selection study: "only chosen genes passed" — unclear how learned selectors are
  trained under this and how fairness is enforced.
- "gene-validation table pending" placeholder; objective coefficients under-justified;
  router z-loss/balancing terms briefly described.
- Missing related work: GexBERT (Jiang & Hassanpour); no efficient-attention baselines.

## Questions for authors (verbatim, condensed)
1. Reconcile Table 2 (PRAD vs THCA), Table 5 (router vs Concrete), Table 4 (expert vs fixed).
2. Define s_i in L_comp; explain computation and gradient flow.
3. How are effective FLOPs computed (which components; aggregation over depth)?
4. Selection study: how is the learned selector trained if "only chosen genes passed"?
5. Total params (incl. embeddings) + end-to-end wall-clock for shared vs independent vs fixed.
6. Multi-seed mean±std on main cohort task + matched standard/efficient-attention transformer.
7. Pancreas: convert genomaps back to gene vectors? Separate on/off-distribution.
8. Sensitivity to temperature schedule + peaked init (isolating ablation).
9. Finalized enrichment (Reactome/GO) + per-class marker lists + significance vs random/variance.
10. Does recursion depth correlate with variance/mean|Z|? Disentangle biology vs statistics.

## Overall
Appealing, novel, biologically motivated; promising. Recommends REVISION before
publication: fix inconsistencies, complete gene validation, add multi-seed +
matched-budget baselines, clarify selection-study training, report total params/timing.
