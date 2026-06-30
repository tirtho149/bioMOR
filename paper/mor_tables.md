# SMART: experiments behind the three claims (genomap + pathway, no TCGA)

macro-F1 (mean±std); `--` = job still running.

## How few tokens suffice — macro-F1 as the number of marker tokens $M$ is reduced (token reduction)
| Dataset / #tokens M | M=16 | M=32 | M=64 | M=128 | M=256 |
|---|---|---|---|---|---|
| tabula_muris | -- | -- | -- | -- | -- |
| pancreas | -- | -- | -- | -- | -- |
| common_class | -- | -- | -- | -- | -- |
| prototype | -- | -- | -- | -- | -- |
| baron | -- | -- | -- | -- | -- |
| segerstolpe | -- | -- | -- | -- | -- |
| lung | -- | -- | -- | -- | -- |
| oesophagus | -- | -- | -- | -- | -- |
| spleen | -- | -- | -- | -- | -- |
| tcell | -- | -- | -- | -- | -- |
| prostate | TODO | TODO | TODO | TODO | TODO |
| blca | TODO | TODO | TODO | TODO | TODO |
| stad | TODO | TODO | TODO | TODO | TODO |
| panmeta_subtype | TODO | TODO | TODO | TODO | TODO |

## Recursion versus independent layers across model sizes — macro-F1 of independent layers (Vanilla), one weight-shared block (Recursive), and adaptive routing (SMART)
| Arch x size (macro_f1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe | lung | oesophagus | spleen | tcell | prostate | blca | stad | panmeta_subtype |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla (independent) small | 77.0 | 63.8 | 71.1 | 96.3 | 72.1 | 7.6 | 74.4 | 60.1 | 51.2 | 58.3 | TODO | TODO | TODO | TODO |
| Vanilla (independent) medium | 80.2 | 52.9 | 70.5 | 95.0 | 71.1 | 7.3 | 74.2 | 60.5 | 51.1 | 47.2 | TODO | TODO | TODO | TODO |
| Vanilla (independent) large | 80.3 | 61.4 | 62.5 | 95.6 | 66.6 | 7.9 | 73.5 | 49.5 | 48.2 | 45.3 | TODO | TODO | TODO | TODO |
| Vanilla (independent) xlarge | 62.0 | 55.2 | 39.9 | 87.8 | 62.6 | 4.0 | 53.9 | 49.8 | 36.0 | 42.1 | TODO | TODO | TODO | TODO |
| Recursive (shared) small | 72.4 | 53.2 | 74.0 | 95.2 | 67.1 | 8.7 | 73.8 | 56.6 | 48.9 | 55.1 | TODO | TODO | TODO | TODO |
| Recursive (shared) medium | 79.2 | 57.1 | 70.3 | 93.8 | 66.2 | 11.1 | 71.7 | 61.8 | 48.3 | 47.5 | TODO | TODO | TODO | TODO |
| Recursive (shared) large | 79.7 | 60.2 | 62.4 | 95.4 | 69.1 | 7.3 | 72.9 | 55.0 | 45.8 | 44.9 | TODO | TODO | TODO | TODO |
| Recursive (shared) xlarge | 53.6 | 53.6 | 36.9 | 94.8 | 50.8 | 5.7 | 54.0 | 41.4 | 39.3 | 44.7 | TODO | TODO | TODO | TODO |
| MoR (SMART) small | 69.3 | 49.4 | 73.7 | 93.1 | 60.3 | 7.3 | 71.9 | 57.8 | 48.4 | 60.5 | TODO | TODO | TODO | TODO |
| MoR (SMART) medium | 77.4 | 49.7 | 71.5 | 94.2 | 61.8 | 6.3 | 71.7 | 61.2 | 49.9 | 48.7 | TODO | TODO | TODO | TODO |
| MoR (SMART) large | 75.1 | 63.0 | 59.7 | 95.4 | 60.4 | 5.9 | 72.8 | 54.9 | 46.9 | 47.6 | TODO | TODO | TODO | TODO |
| MoR (SMART) xlarge | 68.1 | 63.1 | 48.0 | 94.0 | 57.0 | 8.9 | 58.0 | 54.3 | 43.3 | 47.2 | TODO | TODO | TODO | TODO |

## Weight-sharing schemes — Cycle, Sequence, Middle-Cycle and Middle-Sequence sharing between the fully-shared and fully-independent extremes
| Sharing scheme (K=6) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe | lung | oesophagus | spleen | tcell | prostate | blca | stad | panmeta_subtype |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| shared (1 block) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Cycle (3) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Sequence (3) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Middle-Cycle (3) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Middle-Sequence (3) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| independent (6) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

## Where computation is spent — mean recursion depth per marker token and the fraction of tokens still active at each step
| Dataset | mean token depth | active fraction per step (1..K) |
|---|---|---|
| tabula_muris | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| pancreas | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| common_class | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| prototype | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| baron | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| segerstolpe | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| lung | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| oesophagus | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| spleen | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| tcell | 2.75/4 | 1.00, 0.75, 0.50, 0.50 |
| prostate | TODO | TODO |
| blca | TODO | TODO |
| stad | TODO | TODO |
| panmeta_subtype | TODO | TODO |

## Key/value reuse across recursions — recomputing vs reusing the first-step attention keys/values across recursion steps
| KV strategy (macro-F1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe | lung | oesophagus | spleen | tcell | prostate | blca | stad | panmeta_subtype |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| recompute K/V (no cache) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| reuse step-1 K/V (step-cache) | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

## Warm-starting recursion from a fixed-depth model — initialising the shared block from a trained fixed-depth model vs training from scratch
| Warm-start (macro-F1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe | lung | oesophagus | spleen | tcell | prostate | blca | stad | panmeta_subtype |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixed-depth source | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| MoR from scratch | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| MoR warm-started | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| warm-start gain | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |

## Routing configurations — router head, temperature and load balancing for the expert- and token-choice routers
| Routing config (macro-F1) | tabula_muris | pancreas | common_class | prototype | baron | segerstolpe | lung | oesophagus | spleen | tcell | prostate | blca | stad | panmeta_subtype |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| expert linear | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| expert MLP | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| expert temp=2 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| token linear | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| token MLP | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |
| token balance=0.01 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | TODO | TODO | TODO | TODO |

