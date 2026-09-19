#!/bin/bash
# Snapshot finished checkpoint-AUC curves into analysis/results/, then commit and push.
#
# Why this exists: curve results live only in /Warehouse on the training box, which is
# not guaranteed to outlive the batch. This writes them into the repository so the work
# survives the machine.
#
# Usage: snapshot_curves.sh <label> <run> [<run> ...]
set -u
REPO=/home/as_jurabio_com/git/airrtm
M=/Warehouse/Andrei/emerson/model
label=$1; shift
out=$REPO/analysis/results/curves_${label}.md
mkdir -p "$(dirname "$out")"

{
  echo "# checkpoint-AUC curves -- ${label}"
  echo
  echo "Generated $(date -Is) on $(hostname). Untrained bars are NOT applied here;"
  echo "each run must be read against its own random-init control (see FINDINGS.md)."
  echo
  for r in "$@"; do
    echo "## ${r}"
    echo
    f=$M/$r/checkpoint_auc_curve.json
    log=$M/ckptauc_$r.log
    if [ -f "$f" ]; then
      REPO=$REPO python3 - "$f" "$r" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
rows = [r for r in rows if r.get("roc_auc") is not None]
best = max(rows, key=lambda r: r["roc_auc"])
print(f"best: epoch {best['epoch']}  roc_auc={best['roc_auc']:.4f}   "
      f"last: epoch {rows[-1]['epoch']}  roc_auc={rows[-1]['roc_auc']:.4f}   "
      f"({len(rows)} checkpoints)")
print()
print("| epoch | " + " | ".join(str(r["epoch"]) for r in rows) + " |")
print("|---" * (len(rows) + 1) + "|")
print("| test AUC | " + " | ".join(f"{r['roc_auc']:.3f}" for r in rows) + " |")
print("| train AUC | " + " | ".join(f"{r['train_roc_auc']:.3f}" for r in rows) + " |")
PY
    else
      n=$(grep -ac "roc_auc=" "$log" 2>/dev/null || echo 0)
      echo "(no curve json -- ${n} checkpoints scored so far)"
    fi
    echo
  done
} > "$out"

cd $REPO || exit 1
git add -A analysis/results
if git diff --cached --quiet; then
  echo "$(date -Is) snapshot_curves: nothing new to commit"
  exit 0
fi
git commit -q -m "Snapshot checkpoint-AUC curves (${label})

Written by analysis/snapshot_curves.sh so the results survive the training box.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push -q origin v3 && echo "$(date -Is) snapshot_curves: pushed ${label}"
