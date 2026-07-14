# SMART — method content for teaser generation

**Title:** SMART: A Selective Marker-guided Adaptive Recursive Transformer with
Learned Gene-Graph Routing for Single-Cell and Multi-Omics Classification.

## The problem
Transformer models for single-cell / multi-omics classification treat thousands of
genes as equally important tokens and stack many independent layers. This makes them
parameter-heavy (tens–hundreds of millions of params), makes self-attention scale
O(N^2) in the number of genes N, and leaves *which* genes deserve computation entirely
to data. Efficiency is usually recovered only afterward via pruning/distillation.

## The stance
For gene expression, the data and known biology together tell us *where to spend
computation*. (a) A small set of **marker genes** discriminates cell types; (b) gene
co-expression / regulatory networks are approximately scale-free, so a few high-degree
**hub genes** exert outsized influence. So: let the model learn which genes are markers,
and let a gene-gene graph shape the routing decision. Parameter efficiency becomes an
**architectural** property, not a post-hoc fix.

## The method — three coupled components + one biological prior
1. **Marker router (Q-Former-style cross-attention).** M learnable query slots
   cross-attend over ALL N genes with a temperature-annealed softmax (soft -> peaked),
   selecting M interpretable **marker tokens** end-to-end. Gradients still reach every gene.
2. **Marker-driven compression.** Each cell is represented by only its M << N markers,
   cutting attention from O(N^2) to O(M^2). A few dozen tokens recover most full-gene accuracy.
3. **Recursive shared block.** ONE transformer block f_theta applied K times
   (weight-shared, Universal-Transformer / ALBERT style), with a per-marker refinement
   gate between passes and a **Mixture-of-Recursions (MoR)** expert-choice depth router
   that gives each marker its own **adaptive recursion depth** d_m.
4. **Biology-informed router + gene-gene graph (the central finding).** A gene-gene
   graph *smooths* (denoises) the input expression before marker selection, and primes
   the depth router (+ beta_t * pi_m). The graph is either FIXED (co-expression /
   Reactome centrality) or **LEARNED** low-rank end-to-end.

Tokens are mean-pooled over markers and classified with a linear head. The SAME pipeline
serves single-cell expression (learned marker genes) and bulk multi-omics (fixed
**Reactome pathway tokens** pooling mutation + copy-number + expression).

## Headline results (the punchline for a teaser)
- Controlled **none / random-graph / fixed-biology / learned** study: the **LEARNED**
  gene-graph is the decisive positive — **+11.7 points to 70.8% single-cell macro-F1**,
  above a degree-matched **random graph at 57.5%**. A FIXED hand-built prior does NOT
  beat the random-graph control (honest negative).
- A confound factorial isolates the gain as input **smoothing**, not depth routing.
- Efficiency is architectural: **4x parameter reduction** at K=4; **~38% fewer recursion
  FLOPs** at matched accuracy; a few dozen marker tokens recover most full-gene accuracy.
- On raw multi-omics cohorts SMART's multi-modal model is the strongest method and
  matches a far larger pretrained foundation model within noise.
- 11 datasets total (8 single-cell, 3 multi-omics), multi-seed.

## Visual identity of the paper (match this)
- Panel A = **Marker Selection** (green, accentA = HTML 2E7D5B, panel fill = pale green EAF2EA).
- Panel B = **Biology-Informed Recursive Routing & Classification**
  (blue, accentB = HTML 2F5BAA, panel fill = pale blue EAF0FA).
- Rounded rectangular stage boxes, thin grey edges (6B7280), Stealth arrowheads.
- The learned gene-gene graph is drawn in green with a curved dashed "smooth x (denoise)"
  arrow feeding back into the marker router — this is the hero of the figure.
