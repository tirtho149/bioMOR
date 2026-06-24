#!/usr/bin/env python3
"""
Extract (venue, title, github_repo) from the downloaded PDFs.

For each PDF in pdfs/ we:
  * recover its title + venue from papers.jsonl (matched via the same slug rule
    get_pdfs.py used to name the file),
  * scan the PDF for GitHub repository links, looking at BOTH the rendered text
    and the embedded hyperlink (URI) annotations -- most papers encode the repo
    as a clickable link, so text-only scraping misses many of them,
  * normalise each hit to a canonical https://github.com/<owner>/<repo> form and
    pick one primary repo per paper.

Output: repos.csv  with columns  slug,venue,title,github_repo,all_github_repos
"""
import argparse
import csv
import json
import os
import re
import time
from collections import Counter

import fitz  # PyMuPDF
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(HERE, "pdfs")
PAPERS = os.path.join(HERE, "papers.jsonl")
OUT = os.path.join(HERE, "repos.csv")

# github paths that are NOT repos
RESERVED_OWNERS = {
    "about", "pricing", "features", "marketplace", "sponsors", "topics",
    "collections", "explore", "settings", "login", "join", "apps", "orgs",
    "site", "contact", "security", "enterprise", "readme", "search", "notifications",
}

GH_RE = re.compile(r"github\.com/([A-Za-z0-9][\w.-]*)/([A-Za-z0-9][\w.-]*)", re.I)


def slugify_from_paper(paper):
    """Mirror get_pdfs.slugify so we can match papers.jsonl rows to pdf files."""
    ext = paper.get("externalIds") or {}
    base = ext.get("DOI") or ext.get("ArXiv") or paper.get("paperId") or paper.get("title", "untitled")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", str(base)).strip("_")
    return base[:120] or "untitled"


def canon_repo(owner, repo):
    """Canonicalise to https://github.com/owner/repo or None if not a real repo."""
    owner = owner.strip().strip(".")
    repo = repo.strip()
    # strip trailing junk pulled in from prose / .git suffix
    repo = re.sub(r"\.git$", "", repo, flags=re.I)
    repo = repo.rstrip(").,;:'\"]}>")
    if not owner or not repo:
        return None
    if owner.lower() in RESERVED_OWNERS:
        return None
    if repo.lower() in {"blob", "tree", "raw", "wiki", "releases", "issues"}:
        return None
    return f"https://github.com/{owner}/{repo}"


def find_repos(pdf_path):
    """Return a Counter of canonical repo URLs found in the PDF."""
    hits = Counter()
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return hits
    with doc:
        for page in doc:
            # 1) embedded hyperlink annotations (the reliable source)
            for link in page.get_links():
                uri = link.get("uri") or ""
                m = GH_RE.search(uri)
                if m:
                    c = canon_repo(m.group(1), m.group(2))
                    if c:
                        hits[c] += 1
            # 2) rendered text (catches links printed as plain text)
            text = page.get_text("text")
            for m in GH_RE.finditer(text):
                c = canon_repo(m.group(1), m.group(2))
                if c:
                    hits[c] += 1
    return hits


def resolves(session, url, tries=3):
    """True if a GitHub repo URL returns a non-404 status; None on hard failure."""
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=20, allow_redirects=True,
                            headers={"User-Agent": "repo-validator/1.0"})
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(int(r.headers.get("Retry-After", 5)) + 2)
            continue
        return r.status_code < 400
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-links", action="store_true",
                    help="resolve each repo over HTTP and record repo_status (live/dead)")
    args = ap.parse_args()

    # slug -> {title, venue}
    meta = {}
    with open(PAPERS) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            meta[slugify_from_paper(p)] = {
                "title": (p.get("title") or "").strip(),
                "venue": (p.get("venue") or "").strip(),
            }

    pdfs = sorted(f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf"))
    rows = []
    n_with_repo = 0
    for fn in pdfs:
        slug = fn[:-4]
        info = meta.get(slug, {"title": "", "venue": ""})
        hits = find_repos(os.path.join(PDF_DIR, fn))
        # primary = most frequently seen repo (tie -> first inserted)
        primary = hits.most_common(1)[0][0] if hits else ""
        all_repos = ";".join(dict.fromkeys(  # de-dup, preserve order by count
            r for r, _ in hits.most_common()))
        if primary:
            n_with_repo += 1
        rows.append({
            "slug": slug,
            "venue": info["venue"],
            "title": info["title"],
            "github_repo": primary,
            "all_github_repos": all_repos,
            "repo_status": "",
        })

    n_live = n_dead = 0
    if args.check_links:
        sess = requests.Session()
        cache = {}
        for r in rows:
            url = r["github_repo"]
            if not url:
                continue
            if url not in cache:
                ok = resolves(sess, url)
                cache[url] = "live" if ok else ("dead" if ok is False else "unknown")
                time.sleep(0.3)  # be polite to github.com
            r["repo_status"] = cache[url]
        n_live = sum(1 for r in rows if r["repo_status"] == "live")
        n_dead = sum(1 for r in rows if r["repo_status"] == "dead")

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["slug", "venue", "title", "github_repo",
                                           "all_github_repos", "repo_status"])
        w.writeheader()
        w.writerows(rows)

    print(f"[extract_repos] PDFs scanned : {len(rows)}")
    print(f"[extract_repos] with a repo  : {n_with_repo}")
    print(f"[extract_repos] no repo found: {len(rows) - n_with_repo}")
    if args.check_links:
        print(f"[extract_repos] live repos   : {n_live}")
        print(f"[extract_repos] dead repos   : {n_dead}")
    print(f"[extract_repos] wrote        : {OUT}")


if __name__ == "__main__":
    main()
