#!/usr/bin/env bash
# Publish engram-chars/docs to a gh-pages branch (site at branch root).
# Lean: keeps only base.glb per character (animations are off by default), so
# the Pages payload stays small. Uses a git worktree so the working tree is
# never disturbed. Run from anywhere inside the engram repo.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SRC="$REPO_ROOT/engram-chars/docs"
[ -d "$SRC" ] || { echo "cannot find $SRC"; exit 1; }

STAGE="$(mktemp -d)"
WT="$(mktemp -d)"
rsync -a --exclude 'sessions' --exclude '.DS_Store' "$SRC/" "$STAGE/"

# Trim heavy animation GLBs; keep base.glb. Blank manifest animations so the
# frontend loads base.glb (static pose).
python3 - "$STAGE" <<'PY'
import os, sys, json, glob
root = sys.argv[1]
cdir = os.path.join(root, 'assets', 'characters')
if os.path.isdir(cdir):
    for d in glob.glob(os.path.join(cdir, '*')):
        if not os.path.isdir(d): continue
        for g in glob.glob(os.path.join(d, '*.glb')):
            if os.path.basename(g) != 'base.glb':
                os.remove(g)
        mp = os.path.join(d, 'manifest.json')
        if os.path.exists(mp):
            try:
                m = json.load(open(mp)); m['animations'] = {}
                json.dump(m, open(mp, 'w'), indent=2)
            except Exception:
                pass
PY

git -C "$REPO_ROOT" worktree add --force -B gh-pages "$WT" >/dev/null 2>&1 \
  || git -C "$REPO_ROOT" worktree add --force "$WT" gh-pages
find "$WT" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
cp -R "$STAGE"/. "$WT"/
touch "$WT/.nojekyll"     # serve files starting with underscores, skip Jekyll

git -C "$WT" add -A
git -C "$WT" commit -m "Publish engram-chars site to gh-pages" >/dev/null 2>&1 || echo "(no changes)"
git -C "$WT" push -f origin gh-pages

git -C "$REPO_ROOT" worktree remove --force "$WT"
rm -rf "$STAGE"
echo "Published to gh-pages. In GitHub: Settings, Pages, Source = gh-pages branch, / (root)."
