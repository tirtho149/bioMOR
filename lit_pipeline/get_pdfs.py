#!/usr/bin/env python3
"""
Step 2 (THE MAIN THING): download the actual PDF for every harvested paper,
trying ALL available free APIs in turn until one yields a real PDF.

Resolver order (all free, no paid keys):
  1. Semantic Scholar  openAccessPdf.url        (already in the harvest record)
  2. arXiv             externalIds.ArXiv        -> https://arxiv.org/pdf/<id>
  3. Unpaywall         externalIds.DOI          -> best_oa_location.url_for_pdf
  4. OpenAlex          externalIds.DOI          -> best_oa_location.pdf_url
  5. Europe PMC        PMID / PMCID             -> fullTextUrlList / PMC pdf
  6. Crossref          externalIds.DOI          -> link[].URL (pdf intended-application)

Each candidate URL is downloaded and validated by its %PDF magic bytes, so we
never save an HTML paywall page as a ".pdf".

Inputs : papers.jsonl  (from harvest.py)
Outputs: pdfs/<slug>.pdf          one file per successfully fetched paper
         manifest.csv             per-paper status (source used, path, bytes)
Idempotent: re-running skips papers whose PDF already exists.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import requests

UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "tracclaude3@gmail.com")
HEADERS = {"User-Agent": "RecursiveQFormer-LitPipeline/1.0 (research; mailto:%s)" % UNPAYWALL_EMAIL}


# ---------- helpers ---------------------------------------------------------

def slugify(paper):
    """Stable, filesystem-safe id for a paper."""
    ext = paper.get("externalIds") or {}
    base = ext.get("DOI") or ext.get("ArXiv") or paper.get("paperId") or paper.get("title", "untitled")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(base)).strip("_")
    return base[:120] or "untitled"


def looks_like_pdf(content):
    return content[:5] == b"%PDF-"


def try_download(session, url, dest, referer=None, tries=5):
    """GET a URL, save iff it is a real PDF. Retries on 429. Returns True on success."""
    if not url:
        return False
    hdrs = dict(HEADERS)
    if referer:
        hdrs["Referer"] = referer
    for attempt in range(tries):
        try:
            with session.get(url, headers=hdrs, timeout=60, stream=True, allow_redirects=True) as r:
                if r.status_code == 429:           # rate limited (e.g. OpenReview)
                    wait = int(r.headers.get("Retry-After", 0)) or 4 * (attempt + 1)
                    time.sleep(min(wait, 30))
                    continue
                if r.status_code != 200:
                    return False
                ctype = r.headers.get("Content-Type", "").lower()
                chunks, first = [], b""
                for chunk in r.iter_content(8192):
                    if not first:
                        first = chunk
                        # bail early if it's clearly HTML/JSON and not a PDF
                        if not looks_like_pdf(first) and ("pdf" not in ctype):
                            if first.lstrip()[:1] in (b"<", b"{"):
                                return False
                    chunks.append(chunk)
                content = b"".join(chunks)
            if not looks_like_pdf(content):
                return False
            tmp = dest + ".part"
            with open(tmp, "wb") as fh:
                fh.write(content)
            os.replace(tmp, dest)
            return True
        except requests.RequestException:
            return False
    return False


# ---------- per-source URL resolvers ---------------------------------------

def src_s2(session, paper):
    oa = paper.get("openAccessPdf") or {}
    return [oa.get("url")] if oa.get("url") else []


def src_arxiv(session, paper):
    aid = (paper.get("externalIds") or {}).get("ArXiv")
    if not aid:
        return []
    return [f"https://arxiv.org/pdf/{aid}", f"https://arxiv.org/pdf/{aid}.pdf"]


def src_unpaywall(session, paper):
    doi = (paper.get("externalIds") or {}).get("DOI")
    if not doi:
        return []
    try:
        r = session.get(f"https://api.unpaywall.org/v2/{doi}",
                        params={"email": UNPAYWALL_EMAIL}, timeout=30)
        if r.status_code != 200:
            return []
        d = r.json()
        urls = []
        best = d.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            urls.append(best["url_for_pdf"])
        for loc in (d.get("oa_locations") or []):
            if loc.get("url_for_pdf"):
                urls.append(loc["url_for_pdf"])
        return urls
    except (requests.RequestException, ValueError):
        return []


def src_openalex(session, paper):
    doi = (paper.get("externalIds") or {}).get("DOI")
    if not doi:
        return []
    try:
        r = session.get(f"https://api.openalex.org/works/https://doi.org/{doi}",
                        params={"mailto": UNPAYWALL_EMAIL}, timeout=30)
        if r.status_code != 200:
            return []
        d = r.json()
        urls = []
        for key in ("best_oa_location", "primary_location"):
            loc = d.get(key) or {}
            if loc.get("pdf_url"):
                urls.append(loc["pdf_url"])
        for loc in (d.get("locations") or []):
            if loc.get("pdf_url"):
                urls.append(loc["pdf_url"])
        return urls
    except (requests.RequestException, ValueError):
        return []


def src_europepmc(session, paper):
    ext = paper.get("externalIds") or {}
    pmid, pmcid = ext.get("PubMed"), ext.get("PubMedCentral")
    urls = []
    if pmcid:
        pmc = pmcid if str(pmcid).upper().startswith("PMC") else f"PMC{pmcid}"
        urls.append(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextPDF")
    ident = pmcid or pmid
    if ident:
        try:
            q = f"EXT_ID:{pmid}" if pmid else f"PMCID:{pmc}"
            r = session.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                            params={"query": q, "format": "json", "resultType": "core"}, timeout=30)
            if r.status_code == 200:
                for res in (r.json().get("resultList", {}).get("result") or [])[:1]:
                    for u in (res.get("fullTextUrlList", {}).get("fullTextUrl") or []):
                        if u.get("documentStyle") == "pdf" and u.get("url"):
                            urls.append(u["url"])
        except (requests.RequestException, ValueError):
            pass
    return urls


def src_crossref(session, paper):
    doi = (paper.get("externalIds") or {}).get("DOI")
    if not doi:
        return []
    try:
        r = session.get(f"https://api.crossref.org/works/{doi}", timeout=30, headers=HEADERS)
        if r.status_code != 200:
            return []
        links = (r.json().get("message", {}) or {}).get("link") or []
        return [l["URL"] for l in links
                if l.get("URL") and "pdf" in (l.get("content-type", "") + l.get("intended-application", "")).lower()]
    except (requests.RequestException, ValueError):
        return []


RESOLVERS = [
    ("s2_openaccess", src_s2),
    ("arxiv", src_arxiv),
    ("unpaywall", src_unpaywall),
    ("openalex", src_openalex),
    ("europepmc", src_europepmc),
    ("crossref", src_crossref),
]


# ---------- driver ----------------------------------------------------------

def main():
    here = os.path.dirname(__file__)
    ap = argparse.ArgumentParser(description="Download free PDFs for harvested papers (all free APIs).")
    ap.add_argument("--in", dest="inp", default=os.path.join(here, "papers.jsonl"))
    ap.add_argument("--pdf-dir", default=os.path.join(here, "pdfs"))
    ap.add_argument("--manifest", default=os.path.join(here, "manifest.csv"))
    ap.add_argument("--max", type=int, default=0, help="Cap papers processed (0 = all).")
    ap.add_argument("--sleep", type=float, default=0.5, help="Seconds between papers (be polite).")
    args = ap.parse_args()

    os.makedirs(args.pdf_dir, exist_ok=True)
    sess = requests.Session()
    if os.environ.get("S2_API_KEY"):
        sess.headers["x-api-key"] = os.environ["S2_API_KEY"]

    with open(args.inp) as fh:
        papers = [json.loads(l) for l in fh if l.strip()]
    if args.max:
        papers = papers[:args.max]
    print(f"[get_pdfs] {len(papers)} papers to resolve; email={UNPAYWALL_EMAIL}")

    stats = {name: 0 for name, _ in RESOLVERS}
    got = skipped = failed = 0

    with open(args.manifest, "w", newline="") as mf:
        w = csv.writer(mf)
        w.writerow(["slug", "status", "source", "path", "bytes", "title", "year", "doi", "arxiv"])
        for i, p in enumerate(papers, 1):
            slug = slugify(p)
            dest = os.path.join(args.pdf_dir, slug + ".pdf")
            ext = p.get("externalIds") or {}
            row_tail = [p.get("title", "")[:200], p.get("year", ""), ext.get("DOI", ""), ext.get("ArXiv", "")]

            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                skipped += 1
                w.writerow([slug, "skip_exists", "cache", dest, os.path.getsize(dest)] + row_tail)
                print(f"  [{i}/{len(papers)}] skip (have) {slug}")
                continue

            chosen = None
            for name, fn in RESOLVERS:
                for url in fn(sess, p):
                    if try_download(sess, url, dest, referer=p.get("url")):
                        chosen = name
                        break
                if chosen:
                    break

            if chosen:
                got += 1
                stats[chosen] += 1
                sz = os.path.getsize(dest)
                w.writerow([slug, "ok", chosen, dest, sz] + row_tail)
                print(f"  [{i}/{len(papers)}] OK via {chosen:14s} {slug} ({sz//1024} KB)")
            else:
                failed += 1
                w.writerow([slug, "no_pdf", "", "", 0] + row_tail)
                print(f"  [{i}/{len(papers)}] -- no free PDF found for {slug}")
            mf.flush()
            time.sleep(args.sleep)

    print("\n[get_pdfs] DONE")
    print(f"  downloaded : {got}")
    print(f"  cached     : {skipped}")
    print(f"  no pdf     : {failed}")
    print(f"  by source  : " + ", ".join(f"{k}={v}" for k, v in stats.items() if v))
    print(f"  manifest   : {args.manifest}")
    print(f"  pdfs in    : {args.pdf_dir}")


if __name__ == "__main__":
    main()
