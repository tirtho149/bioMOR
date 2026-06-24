# Gene-Classification Literature Pipeline

Collect **gene / genomic classification** papers from **ICLR, ICML, AAAI (2024–2026)**
and download their **PDFs** — using only **free, no-key APIs**. Image-/vision-based
papers are excluded (sequence/expression work only).

The headline deliverable is **`pdfs/`** — the actual PDF of every paper we can
legally get for free.

## Pipeline

```
harvest_openalex.py   ──┐
                        ├─►  papers.jsonl  ──►  get_pdfs.py  ──►  pdfs/*.pdf + manifest.csv
harvest_openreview.py ──┘
```

1. **Harvest metadata** (build the worklist). Two sources, because no single API
   covers all three venues well:
   - `harvest_openalex.py`  → **AAAI** (OpenAlex indexes it fully; gives DOI + pdf_url).
   - `harvest_openreview.py` → **ICLR & ICML** (these live on OpenReview, which OpenAlex
     barely indexes; OpenReview gives the PDF link directly).
   - `harvest.py` is a third, generic **Semantic Scholar bulk** harvester (any venue,
     any topic) kept for ad-hoc queries — not used in the default ICLR/ICML/AAAI run.

   Both write the **same JSONL schema** so the downloader is source-agnostic.

2. **Download PDFs** — `get_pdfs.py` is the core. For each paper it tries every free
   source in order until one yields a valid PDF (verified by `%PDF` magic bytes, so a
   paywall HTML page is never saved as `.pdf`):

   | order | source        | key needed | used for                          |
   |-------|---------------|------------|-----------------------------------|
   | 1 | Semantic Scholar `openAccessPdf` | no | direct link already in the record (incl. OpenReview) |
   | 2 | arXiv         | no | `externalIds.ArXiv` → `arxiv.org/pdf/<id>` |
   | 3 | Unpaywall     | email only | `externalIds.DOI` → best OA location |
   | 4 | OpenAlex      | no | `externalIds.DOI` → `pdf_url` |
   | 5 | Europe PMC    | no | `PubMed`/`PMCID` → full-text PDF |
   | 6 | Crossref      | no | `externalIds.DOI` → publisher PDF link |

   429s (e.g. OpenReview rate limiting) are retried with backoff. Re-running is
   idempotent — already-downloaded PDFs are skipped.

## Run it

```bash
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
cd /work/mech-ai-scratch/tirtho/RecusrsiveQFormer/lit_pipeline

# 1. harvest (AAAI via OpenAlex, then append ICLR/ICML via OpenReview)
python harvest_openalex.py   --years 2024-2026 --venues AAAI
python harvest_openreview.py --years 2024,2025,2026 --venues ICLR,ICML --append

# 2. download every free PDF
python get_pdfs.py --sleep 1.2
```

Outputs:
- `papers.jsonl` — one record per paper (title, year, venue, authors, abstract,
  externalIds, openAccessPdf, …).
- `pdfs/<slug>.pdf` — the downloaded PDFs.
- `manifest.csv` — per-paper status: `ok` / `no_pdf` / `skip_exists`, which source
  succeeded, byte size, title, year, DOI, arXiv id.

## Relevance filtering (`gene_filter.py`)

A paper is kept only if its title/abstract contains an **unambiguous genomic** token
(`gene`, `dna`, `rna`, `genomic`, `transcriptom`, `scRNA`, `gene expression`,
`single-cell`, …). Deliberately **excluded**: `variant` / `mutation` — these are also
ML jargon ("model variant", evolutionary-algorithm "mutation") and flooded the set with
point-cloud / time-series papers. Image-/vision-based papers are dropped too.

## Free-API notes

- **No API keys required.** Set `S2_API_KEY` (optional) for a higher Semantic Scholar
  rate limit; set `UNPAYWALL_EMAIL` (defaults to the project email) for Unpaywall/OpenAlex
  polite-pool access.
- All requests send a descriptive User-Agent with a contact email (polite-pool etiquette).
