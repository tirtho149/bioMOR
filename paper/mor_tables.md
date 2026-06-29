# SMART: experiments behind the three claims (genomap + pathway, no TCGA)

macro-F1 (mean±std); `--` = job still running.

## How few tokens suffice — macro-F1 as the number of marker tokens $M$ is reduced (token reduction)
| Dataset / #tokens M | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|
| tabula_muris | 36.3 | 53.2 | 65.2 | 70.9 | 76.8 |
| pancreas | 36.1 | 50.3 | 54.0 | 59.0 | 57.2 |
| common_class | 14.7 | 30.9 | 33.9 | 64.8 | 74.9 |
| prototype | 63.2 | 79.1 | 87.4 | 93.8 | 97.5 |
| baron | 28.1 | 43.3 | 42.1 | 61.9 | 72.0 |
| segerstolpe | 5.1 | 6.1 | 10.0 | 4.5 | 7.5 |

## Recursion versus independent layers across model sizes — macro-F1 of independent layers (Vanilla), one weight-shared block (Recursive), and adaptive routing (SMART)
| Arch x size (macro_f1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe |
|---|---|---|---|---|---|---|
| Vanilla (independent) d=48 | 74.5 | 51.1 | 78.8 | 95.5 | 69.8 | 7.3 |
| Vanilla (independent) d=96 | 78.0 | 55.5 | 71.7 | 94.7 | 64.5 | 9.8 |
| Vanilla (independent) d=192 | 78.0 | 56.8 | 3.0 | 96.2 | 3.3 | 9.2 |
| Vanilla (independent) d=384 | 52.4 | 5.3 | 3.0 | 76.3 | 3.3 | 5.9 |
| Recursive (shared) d=48 | 73.7 | 54.3 | 77.6 | 94.3 | 67.0 | 6.6 |
| Recursive (shared) d=96 | 73.5 | 58.1 | 70.6 | 94.2 | 65.6 | 8.6 |
| Recursive (shared) d=192 | 79.7 | 61.5 | 62.1 | 95.2 | 59.7 | 7.0 |
| Recursive (shared) d=384 | 0.5 | 53.8 | 23.0 | 74.2 | 16.4 | 6.4 |
| MoR (SMART) d=48 | 68.7 | 57.8 | 75.6 | 93.5 | 67.6 | 4.8 |
| MoR (SMART) d=96 | 71.9 | 56.7 | 68.6 | 93.3 | 60.3 | 5.9 |
| MoR (SMART) d=192 | 74.0 | 65.6 | 61.5 | 95.3 | 62.4 | 10.0 |
| MoR (SMART) d=384 | 65.8 | 53.4 | 42.8 | 91.6 | 60.1 | 9.9 |

## Weight-sharing schemes — Cycle, Sequence, Middle-Cycle and Middle-Sequence sharing between the fully-shared and fully-independent extremes
| Sharing scheme (K=6) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe | prostate | blca | stad | panmeta_subtype |
|---|---|---|---|---|---|---|---|---|---|---|
| shared (1 block) | 75.1 | 61.0 | 68.9 | 93.6 | 60.2 | 3.4 | 72.3 | 42.7 | 38.8 | -- |
| Cycle (3) | 80.5 | 54.4 | 71.6 | 95.2 | 70.5 | 6.9 | 57.1 | 40.4 | -- | -- |
| Sequence (3) | 78.5 | 57.6 | 0.8 | 93.9 | 66.1 | 4.5 | 59.5 | 43.8 | -- | -- |
| Middle-Cycle (3) | 80.1 | 56.4 | 67.0 | 94.9 | 61.0 | 7.8 | 60.5 | 40.4 | -- | -- |
| Middle-Sequence (3) | 80.1 | 56.4 | 67.0 | 94.9 | 61.0 | 7.8 | 60.5 | 40.4 | -- | -- |
| independent (6) | 79.9 | 57.0 | 3.0 | 95.4 | 65.3 | 7.2 | 59.6 | 44.4 | 56.2 | -- |

## Where computation is spent — mean recursion depth per marker token and the fraction of tokens still active at each step
| Dataset | mean token depth | active fraction per step (1..K) |
|---|---|---|
| tabula_muris | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| pancreas | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| common_class | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| prototype | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| baron | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| segerstolpe | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |

## Key/value reuse across recursions — recomputing vs reusing the first-step attention keys/values across recursion steps
| KV strategy (macro-F1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe | prostate | blca | stad | panmeta_subtype |
|---|---|---|---|---|---|---|---|---|---|---|
| recompute K/V (no cache) | 79.9 | 54.1 | 70.4 | 93.9 | 65.6 | 8.6 | 74.7 | 43.3 | 36.4 | -- |
| reuse step-1 K/V (step-cache) | 71.6 | 58.5 | 71.8 | 93.2 | 59.6 | 6.4 | 82.8 | 39.6 | 38.8 | -- |

## Warm-starting recursion from a fixed-depth model — initialising the shared block from a trained fixed-depth model vs training from scratch
| Warm-start (macro-F1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe |
|---|---|---|---|---|---|---|
| fixed-depth source | 75.2 | 51.8 | 69.0 | 92.9 | 68.1 | 8.3 |
| MoR from scratch | 71.7 | 62.8 | 65.8 | 95.5 | 61.5 | 5.4 |
| MoR warm-started | 65.8 | 61.6 | 64.8 | 93.3 | 55.8 | 6.9 |
| warm-start gain | -5.9 | -1.2 | -0.9 | -2.2 | -5.6 | +1.5 |

## Routing configurations — router head, temperature and load balancing for the expert- and token-choice routers
| Routing config (macro-F1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe |
|---|---|---|---|---|---|---|
| expert linear | 71.9 | 56.7 | 68.6 | 93.3 | 60.3 | 5.9 |
| expert MLP | 67.2 | 51.9 | 70.2 | 93.2 | 49.4 | 7.1 |
| expert temp=2 | 73.5 | 57.6 | 66.7 | 94.1 | 66.2 | 6.5 |
| token linear | 74.7 | 53.7 | 70.1 | 94.5 | 52.6 | 5.4 |
| token MLP | 78.8 | 59.7 | 64.4 | 94.0 | 60.9 | 5.5 |
| token balance=0.01 | 75.6 | 57.6 | 67.5 | 93.7 | 58.6 | 5.0 |

