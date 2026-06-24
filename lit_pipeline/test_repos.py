#!/usr/bin/env python3
"""
Validation tests for repos.csv (the venue/title/github_repo extract).

Run structural checks only (fast, offline):
    pytest test_repos.py -v

Also validate that every GitHub link resolves (hits the network):
    RUN_NETWORK=1 pytest test_repos.py -v
"""
import csv
import os
import re
import time

import pytest
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "repos.csv")
PDF_DIR = os.path.join(HERE, "pdfs")

CANON_RE = re.compile(r"^https://github\.com/[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
EXPECTED_COLS = ["slug", "venue", "title", "github_repo", "all_github_repos", "repo_status"]


@pytest.fixture(scope="module")
def rows():
    with open(CSV) as fh:
        return list(csv.DictReader(fh))


# ---------- structural / format checks (always run) ------------------------

def test_csv_exists_and_nonempty(rows):
    assert os.path.exists(CSV), "repos.csv missing -- run extract_repos.py"
    assert len(rows) > 0


def test_header_columns(rows):
    assert list(rows[0].keys()) == EXPECTED_COLS


def test_one_row_per_pdf(rows):
    n_pdf = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")])
    assert len(rows) == n_pdf, f"{len(rows)} csv rows vs {n_pdf} pdfs on disk"


def test_every_row_has_title_and_venue(rows):
    bad = [r["slug"] for r in rows if not r["title"].strip() or not r["venue"].strip()]
    assert not bad, f"rows missing title/venue: {bad}"


def test_github_links_are_canonical(rows):
    bad = []
    for r in rows:
        for url in filter(None, [r["github_repo"]] + r["all_github_repos"].split(";")):
            if not CANON_RE.match(url):
                bad.append((r["slug"], url))
    assert not bad, f"non-canonical github urls: {bad[:10]}"


def test_primary_repo_is_in_all_repos(rows):
    bad = [r["slug"] for r in rows
           if r["github_repo"] and r["github_repo"] not in r["all_github_repos"].split(";")]
    assert not bad, f"primary repo not listed in all_github_repos: {bad}"


def test_repo_status_values(rows):
    """A row has a status iff it has a repo, and the value is from a fixed set."""
    ok = {"live", "dead", "unknown"}
    for r in rows:
        if r["github_repo"]:
            assert r["repo_status"] in ok, f"{r['slug']}: bad status {r['repo_status']!r}"
        else:
            assert r["repo_status"] == "", f"{r['slug']}: status set but no repo"


# ---------- liveness check (network) ---------------------------------------

def _resolves(session, url, tries=3):
    """True if the GitHub repo URL returns a non-404 status."""
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
    return False


@pytest.mark.skipif(os.environ.get("RUN_NETWORK") != "1",
                    reason="set RUN_NETWORK=1 to verify recorded repo_status against the live web")
def test_recorded_status_is_truthful(rows):
    """Every repo marked 'live' must actually resolve; flag any 'dead' that came back.

    This validates that repo_status in the CSV reflects reality, rather than
    failing the build over repos the authors simply never published.
    """
    sess = requests.Session()
    live = sorted({r["github_repo"] for r in rows if r["repo_status"] == "live"})
    dead = sorted({r["github_repo"] for r in rows if r["repo_status"] == "dead"})

    lying_live = []   # marked live but does not resolve  -> CSV is wrong
    revived = []      # marked dead but now resolves      -> CSV is stale
    for u in live:
        if not _resolves(sess, u):
            lying_live.append(u)
        time.sleep(0.3)
    for u in dead:
        if _resolves(sess, u):
            revived.append(u)
        time.sleep(0.3)

    assert not lying_live, f"marked 'live' but unreachable: {lying_live}"
    assert not revived, f"marked 'dead' but now resolves (re-run --check-links): {revived}"
