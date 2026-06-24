# Baseline reproduction plan — SMART (foundation/deep baselines)

Goal: reproduce ALL 7 deep/foundation baselines on the genoNet tasks under the
SAME seed-42 stratified 80/20 split as SMART, fill `tab:foundation` in the paper.
Honesty rule: never fabricate a baseline F1; pending cells show `--`.

## Per-method protocol (each uses its native, fair setup)
| Baseline | Protocol | Env | Adapter | sbatch |
|---|---|---|---|---|
| scGeneFit | LP marker select + NearestCentroid (CPU) | main `.venv` | dl_baselines.py `--only scGeneFit` | n/a (CPU) |
| Geneformer | fine-tune V2-104M_CLcancer (HF Trainer) | `.venv_geneformer` (transformers 4.44) | run_geneformer.py | run_geneformer.sbatch |
| sciLaMA | beta-VAE embed + logistic probe | `.venv_scilama` | run_scilama.py | run_scilama.sbatch |
| scGPT | frozen embed (pan-cancer) + logistic probe | `.venv_scgpt` | run_scgpt.py (TODO) | run_scgpt.sbatch |
| CellPLM | zero-shot embed (85M) + logistic probe | `.venv_cellplm` (TODO) | run_cellplm.py | run_cellplm.sbatch (TODO) |
| Cell2Sentence | frozen embed (Pythia-410M) + probe | `.venv_c2s` (TODO) | run_c2s.py (TODO) | TODO |
| scbenchmark | train 6L/256d transformer from scratch | `.venv_scbench` (TODO) | run_scbench.py (TODO) | TODO |

## Status (2026-06-24)
- DONE: scGeneFit (mean 44.0 vs SMART 46.5).
- DONE: Geneformer (job 11207472) -> Geneformer.json (stage 46.2, T 41.3, N 34.1, OS 53.5, tumor 54.2).
- QUEUED (all 5 remaining, a100-pinned, seed-42): scGPT 11209667, sciLaMA 11209668,
  scbenchmark 11209669, Cell2Sentence 11209670, CellPLM 11209673.
- All adapters written (run_scgpt.py, run_c2s.py, run_scbench.py + existing). All envs built:
  .venv_{scgpt,scilama,scbench,c2s,cellplm} (torch 2.12.1+cu130; c2s transformers 5.12.1).

### Fixes applied this round (re-apply if rebuilding)
- ALL baseline sbatch MUST pin `--gres=gpu:a100:1`. Generic `gpu:1` lands on nodes whose
  compute capability the cu128/cu130 torch build lacks -> "CUDA error: no kernel image is
  available for execution on the device". a100 (sm_80) is covered (Geneformer used it).
- PyTorch-Lightning jobs (sciLaMA) must `export SLURM_JOB_NAME=bash` before python, else
  Lightning binds the SLURM cluster env and aborts over SLURM_NTASKS.
- scGPT: import the LOCAL repo scgpt (sys.path FIRST), not pip wheel 0.2.4 (wheel hard-imports
  torchtext 0.18, ABI-incompatible with torch 2.11+; repo copy has a vocab_compat shim). Env
  needs `ipython`; call embed_data(use_fast_transformer=False) (no flash_attn).
- scbenchmark: no bulk-TCGA pretrained ckpt, so run_scbench.py trains scModel (6L/256d) + linear
  head FROM SCRATCH end-to-end per task (their quantile binning + <cls> cell emb).

## Execution order
1. Geneformer finishes -> paper auto-refresh.
2. sciLaMA + scGPT envs finish -> `sbatch run_scilama.sbatch`, write run_scgpt.py, `sbatch run_scgpt.sbatch`.
3. Build CellPLM/Cell2Sentence/scbenchmark envs (staggered, not 5 parallel torch installs), write remaining adapters, submit.
4. Each sbatch ends by running make_paper -> tab:foundation fills row-by-row.

## Key facts
- GPU partition = nova (a100/h200/v100). No conda. git-lfs absent (use huggingface_hub).
- Data: data/tcga/unified_bio5.csv, genes `SYMBOL|entrez`, log2-RSEM, 2738 samples.
- Gene map: data/geneformer_aux/entrez2ensembl_human.json + symbol->Ensembl (Geneformer dict) -> 80% coverage; Geneformer pseudo-counts = 2^v-1.
- Results: results_dl_baselines/<method>.json (schema: {method, tasks:{task:{accuracy,macro_f1,weighted_f1,...}}}).
- params.json holds real param counts (SMART 712457; scGPT 51.3M, CellPLM 82.4M, Geneformer 104.4M, C2S 405.3M).
- Paper table generator: make_paper.foundation_baselines_table() -> @@FOUNDATION_TABLE@@ (Model·Venue·#Params·×SMART·Macro-F1).
