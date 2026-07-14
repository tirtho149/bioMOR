#!/usr/bin/env bash
# Bidirectional near-real-time sync between this repo's paper/ dir and an Overleaf
# project (via Overleaf's Git bridge). Polls every $INTERVAL seconds:
#   1. pulls Overleaf edits  -> paper/   (*.tex / *.bib prose the user edits online)
#   2. pushes HPC changes    -> Overleaf (regenerated .tex tables, figs, source)
# No inotify needed (not available on this cluster); 3s poll = effectively live.
#
# One-time setup (run yourself so the token is entered interactively, never stored
# in this script):
#   git config --global credential.helper store
#   git clone https://git.overleaf.com/6a55bd5d7ac6e00e0c59771d \
#       /work/mech-ai-scratch/tirtho/RecusrsiveQFormer/.overleaf_sync
#   (username = git ,  password = your Overleaf Git token)
#
# Then launch durably:
#   nohup bash overleaf_sync.sh > logs/overleaf_sync.log 2>&1 &
set -uo pipefail

PROJ="/work/mech-ai-scratch/tirtho/RecusrsiveQFormer"
PAPER="$PROJ/paper"
SYNC="$PROJ/.overleaf_sync"
PROJECT_ID="${OVERLEAF_PROJECT_ID:-6a55bd5d7ac6e00e0c59771d}"
INTERVAL="${OVERLEAF_SYNC_INTERVAL:-3}"

if [ ! -d "$SYNC/.git" ]; then
  echo "[sync] ERROR: $SYNC is not a git clone. Do the one-time clone first (see header)." >&2
  exit 1
fi
cd "$SYNC" || exit 1
git config pull.rebase true
git config rebase.autoStash true

log(){ echo "[sync $(date -u +%FT%TZ)] $*"; }

# source-only rsync (checksum => identical content never re-transfers, so no ping-pong)
push_to_overleaf(){   # paper/ -> Overleaf clone  (do NOT --delete: keep Overleaf-only files)
  # -L dereferences symlinks: paper/figs/*_cv5.png are symlinks to regenerated
  # figures on the HPC; Overleaf needs the real image bytes, not a dangling link.
  rsync -rLt --checksum \
    --include='*/' \
    --include='*.tex' --include='*.bib' --include='*.cls' --include='*.sty' \
    --include='*.bst' --include='*.bbl' \
    --include='figs/***' \
    --exclude='main.pdf' \
    --exclude='*' \
    "$PAPER/" "$SYNC/"
}
pull_to_paper(){      # Overleaf clone -> paper/  (only prose the user edits online)
  rsync -rlt --checksum \
    --include='*/' --include='*.tex' --include='*.bib' \
    --exclude='.git/' --exclude='*' \
    "$SYNC/" "$PAPER/"
}

log "watching $PAPER <-> Overleaf project $PROJECT_ID every ${INTERVAL}s"
while true; do
  # 1. remote -> local
  if git fetch -q origin 2>/dev/null; then
    if [ "$(git rev-parse @)" != "$(git rev-parse @{u} 2>/dev/null)" ]; then
      if git pull --rebase --autostash -q; then
        pull_to_paper
        log "pulled Overleaf edits into paper/"
      else
        git rebase --abort 2>/dev/null
        log "WARN pull/rebase conflict; skipping this cycle"
      fi
    fi
  else
    log "WARN fetch failed (network/auth?)"
  fi

  # 2. local -> remote
  push_to_overleaf
  if [ -n "$(git status --porcelain)" ]; then
    git add -A
    git commit -q -m "auto-sync from HPC $(date -u +%FT%TZ)"
    git pull --rebase --autostash -q 2>/dev/null
    if git push -q; then
      log "pushed HPC changes to Overleaf"
    else
      log "WARN push failed; will retry next cycle"
    fi
  fi

  sleep "$INTERVAL"
done
