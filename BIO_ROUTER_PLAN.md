# Biology-Informed Router (genomap gene-gene interaction prior) — Plan

> New component for SMART. Status legend: [x] done · [~] in progress · [ ] todo.
> Origin: shared idea (claude.ai) = a *biology-informed router*; user instruction =
> "take genomap's gene-gene interaction identification approach and add it in the
> router classifier, then do proper ablations to show its importance."

## 1. Idea (one line)
Move biology from a post-hoc *validation* step into the routing *decision*: add an
annealed additive **biological prior** to the expert-choice depth-router logit, where
the prior is the **network centrality** of each gene in genomap's gene-gene
co-expression graph.

## 2. Math
Expert-choice depth router, marker token `h_m` at recursion step `t`:

    r_m^(t)   =  (w_r^T h_m^(t)) / tau                      (data-driven, learned)
    r~_m^(t)  =  r_m^(t)  +  beta_t * pi_m                  (biology-informed)
    keep m if r~_m^(t) in top-ceil(c_t M);  gate g = sigmoid(r~) * alpha

- `pi_m` = z-scored **eigenvector centrality** of the gene-gene graph `W`
  (genomap interaction). Hubs (master regulators / co-expression hubs) get a higher
  prior, so they are nudged to recurse deeper.
- `beta_t = beta_0 * (1 - progress)` — a **warm-start** prior that decays to 0, so
  biology guides early (when hidden states are still random — the regime where the
  paper's uniform-init routers fail) and the data-driven term takes over late. Same
  logic as the marker-router temperature anneal.
- Differentiable-handle preserved: `pi_m` is a constant additive bias; `r~` is still
  smooth in `w_r`; the sigmoid gate still multiplies the block output, so gradient
  still flows to the router. Hard top-k still does the discrete keep/drop.

## 3. The prior from genomap (gene-gene interaction identification)
genomap (`genomap/genomap/genomap.py::createInteractionMatrix`) defines gene-gene
interaction as `pairwise_distances(data.T, metric='correlation')`. We reuse only that
interaction-identification step:

1. `correlation(X_train)` — Pearson correlation across **train** samples (BLAS, fast).
2. affinity `= |corr|`, sparsify to each gene's top-k neighbours, symmetrise.
3. `pi = z( eigenvector_centrality(W) )` via power iteration.
4. (optional) GCN operator `D^-1/2 (W+I) D^-1/2` for future logit Laplacian smoothing.

**Leakage-safe:** the graph uses **expression only, no labels**, so the prior injects
network structure without leaking cohort labels (unlike a CellMarker/PanglaoDB
marker-odds prior, which the shared chat flagged). This is the honest, defensible
prior.

## 4. Implementation map
- [x] `recursive_marker_transformer/interaction.py` — correlation -> kNN graph ->
  centrality prior (+ operator); modes `coexpr` / `random` / `none`.
- [x] `config.py` — `gene_interaction`, `interaction_knn`, `router_prior_beta`,
  `router_prior_anneal`.
- [x] `router.py` — `ExpertChoiceRouter` (and `TokenChoiceRouter`) accept
  `prior`,`prior_weight`; add `prior_weight * pi` to logits each step.
- [x] `recursion.py` — thread `prior`,`prior_weight` to the router.
- [x] `model.py` — `gene_centrality` buffer, `set_gene_interaction`, gather
  `pi[marker_idx]`, anneal `beta_t` in `set_anneal`, pass to stack.
- [x] `train.py` (cohort) + `genonet_tasks.py` (hard tasks) — build the prior on the
  train split (label-free) and install it.
- [x] `interaction_experiments.py` — ablation runner.
- [x] `run_interaction.sbatch` — GPU job (11180929, 36/36 cells, rc=0).
- [x] interaction matrix sourced from genomap's own `createInteractionMatrix`
  (loaded from genomap source, OT import stubbed; numerically identical to the
  BLAS `|corr|`, so job results stay valid).
- [x] `make_paper.py` — Method subsection, Fig.1 update, theory appendix, ablation
  table + section, abstract/contributions/discussion/conclusion; PDF rebuilt (14 pp,
  0 gaps).

## RESULT (2026-06-23, job 11180929, 3 seeds)
| prior | Cohort | Stage | T | N |
|---|---|---|---|---|
| none | 96.97±1.42 | 45.65±3.00 | 40.76±5.07 | 30.96±3.61 |
| coexpr | 95.63±0.55 | 46.78±1.71 | 40.05±2.42 | 30.92±5.02 |
| random | 95.14±1.33 | 46.86±3.45 | 40.44±1.27 | 31.57±3.16 |

**Honest negative result:** all three priors within one std on every task; coexpr does
NOT separate from the random-graph control, so no gain is attributable to biological
co-expression structure on these labels. Mild variance reduction under coexpr on
stage/T (shrinkage), but no accuracy gain. Reported transparently in the paper.

## 5. Ablation (to show the component's importance)
Modes per task, multi-seed, macro-F1 mean±std:
- **none**   — original SMART router (baseline).
- **coexpr** — real genomap correlation-graph centrality prior (proposed).
- **random** — degree-matched random-graph control (same sparsity, shuffled edges).

Claim it supports: `coexpr > random ≈ none` would show it is the *real co-expression
structure*, not "any bias / any smoothing", that helps.

Tasks:
- **cohort** (4-way, n_hvg=4000) — headline dataset, but near-saturated (~98%), so
  expect small/!separable margins (state honestly).
- **hard genoNet** (pathologic_stage / T / N, all ~20.5k genes) — the low-signal
  regime where the shared-chat framing says a prior is *most* defensible. Primary
  place to look for a real effect.

Optional sweeps: `beta_0 in {0.5,1,2}`, anneal on/off, `knn in {8,16,32}`.

## 6. Honest framing for the paper
Do **not** claim "biology-informed routing boosts accuracy everywhere." Claim: it is a
principled, label-free prior that stabilises routing and improves compute allocation
**when the task signal is weak** (stage/node), annealing out of the way when the data
is informative. This matches the paper's own multi-seed result (learned selection only
beats random when the label carries a marker signal). Report `coexpr` vs `random` vs
`none`; if margins are within noise, say so.

## 7. Run commands
```bash
# ablation (cohort + hard tasks, modes none/coexpr/random, seeds 0-2)
python -m recursive_marker_transformer.interaction_experiments \
    --tasks cohort pathologic_stage pathologic_T pathologic_N \
    --modes none coexpr random --seeds 0 1 2 --out results_interaction
# or on GPU:  sbatch run_interaction.sbatch
# then fold into the paper:
python -m recursive_marker_transformer.make_paper --results results --outdir paper
```
