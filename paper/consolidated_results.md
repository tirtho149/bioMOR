# SMART -- consolidated results (one table)

Each cell is **accuracy / macro-F1±std** (%, mean over seeds). Single-cell: 3 seeds. Cohorts: multi-seed where available, else single-seed.

| Dataset | Bio-informed router | General router (no prior) | Adaptive depth (MoR) | Fixed depth | No recursion (K=1) | Vanilla transformer |
|---|---|---|---|---|---|---|
| Tabula Muris | 81.1±1.3 / 69.9±1.5 | 81.4±2.3 / 71.5±3.0 | 82.5±0.3 / 72.7±1.1 | 85.3±1.8 / 77.1±2.1 | 82.1±1.7 / 73.1±2.4 | 84.7±1.3 / 75.5±0.8 |
| Pancreas | 91.9±1.7 / 59.1±4.1 | 90.9±2.1 / 54.6±1.1 | 88.1±4.0 / 53.5±1.1 | 88.9±6.3 / 53.6±1.6 | 90.9±2.9 / 51.7±5.2 | 91.4±2.2 / 57.3±1.9 |
| Common | 84.1±1.2 / 70.2±2.2 | 81.0±1.5 / 66.2±2.9 | 81.0±1.5 / 66.2±2.9 | 82.9±0.4 / 68.5±2.7 | 81.1±2.1 / 66.2±3.7 | 81.3±2.1 / 66.6±4.4 |
| Prototype | 94.5±0.1 / 94.0±0.2 | 94.7±0.3 / 94.2±0.4 | 94.7±0.3 / 94.2±0.4 | 94.7±0.2 / 94.1±0.3 | 93.8±0.2 / 93.2±0.5 | 94.7±0.2 / 94.2±0.4 |
| Baron | 77.8±7.3 / 57.1±7.3 | 80.6±4.9 / 61.5±1.8 | 84.8±4.8 / 61.1±1.3 | 86.4±2.3 / 66.2±1.8 | 81.9±0.8 / 56.2±2.6 | 78.6±8.6 / 58.8±4.5 |
| Segerstolpe | 17.4±2.1 / 8.4±1.4 | 13.9±2.5 / 5.7±0.6 | 20.0±6.5 / 7.7±0.8 | 17.3±1.1 / 8.1±0.7 | 18.9±2.2 / 10.1±0.8 | 14.9±0.6 / 6.7±1.6 |
| Lung | 79.6±0.5 / 72.6±1.6 | 78.3±2.0 / 70.7±4.2 | 78.3±2.0 / 70.7±4.2 | 77.6±1.6 / 70.3±1.9 | 78.5±1.8 / 70.3±1.3 | 79.8±0.2 / 72.5±2.4 |
| Oesophagus | 80.7±2.0 / 56.5±5.0 | 80.6±1.4 / 56.1±3.8 | 80.6±1.4 / 56.1±3.8 | 81.0±0.2 / 55.1±1.3 | 79.3±1.8 / 56.5±4.0 | 81.3±0.7 / 55.8±2.9 |
| Spleen | 56.3±3.0 / 48.5±2.4 | 58.9±1.1 / 50.4±2.3 | 58.9±1.1 / 50.4±2.3 | 53.2±2.7 / 46.7±3.5 | 56.0±1.1 / 48.8±2.7 | 56.6±2.4 / 49.4±3.6 |
| T-cell | 65.4±2.5 / 48.7±1.4 | 69.2±2.3 / 50.0±3.2 | 67.6±1.5 / 49.7±2.0 | 65.2±1.4 / 49.3±1.8 | 67.3±1.7 / 50.4±1.9 | 65.8±4.6 / 47.7±2.4 |
| Prostate | 66.0±4.5 / 55.0±9.0 | 68.0±3.1 / 54.3±9.3 | 68.6±2.4 / 55.5±8.7 | 62.2±3.0 / 58.3±2.4 | 61.6±2.4 / 50.4±5.9 | 67.8±5.7 / 56.8±12.3 |
| BLCA | 59.7±7.3 / 41.6±3.2 | 59.3±5.6 / 41.8±4.5 | 58.8±7.6 / 41.1±2.9 | 62.6±3.1 / 41.3±2.9 | 55.6±16.6 / 36.0±8.4 | 66.7±1.0 / 40.0±0.4 |
| STAD | 46.6±8.8 / 45.0±7.5 | 50.6±6.0 / 45.4±7.3 | 48.2±8.1 / 44.7±7.7 | 53.8±5.4 / 49.4±3.5 | 45.8±2.0 / 40.7±2.0 | 51.8±4.3 / 47.8±3.5 |
| PanCan | 62.0±4.9 / 57.9±5.9 | 62.1±6.9 / 57.6±6.9 | 63.0±4.3 / 58.4±4.0 | 73.0±2.5 / 69.4±2.3 | 64.2±2.0 / 59.9±1.9 | 56.3±6.5 / 52.1±8.7 |

## Statistical significance analysis
- Biological prior (co-expression vs degree-matched random-graph control, paired over all dataset x seed): mean Delta-F1 = +0.28 pts, 95% CI [-1.2, +1.7], p = 0.871, Cohen's d_z = +0.07; 0/10 datasets significant after Holm-Bonferroni. Verdict: NOT significant -- statistically indistinguishable from a random graph.
- Adaptive depth: recursion vs single pass (K=1) Delta-F1 = +0.60 (p = 0.337); adaptive vs fixed depth is not formally equivalent at a 1.0-pt margin (TOST p = 0.278); mean compute saving 31% (p = 0.000 that saving > 0). Verdict: the compute reduction is the decisive, significant effect; accuracy/depth gains are modest.
