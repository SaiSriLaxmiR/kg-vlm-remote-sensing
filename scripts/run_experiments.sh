#!/usr/bin/env bash
# scripts/run_experiments.sh
# Run full ablation study across datasets and shot settings.

set -e

DATASETS=(resisc45 ucm)
SHOTS=(1 2 4 8 16)
SEEDS=(42 1 2)

echo "════════════════════════════════════════════════"
echo "  KG-APPLeNet Ablation Experiments"
echo "════════════════════════════════════════════════"

run_exp() {
  local dataset=$1
  local shot=$2
  local sem=$3
  local vis=$4
  local seed=$5
  local tag="sem${sem}_vis${vis}"

  echo ""
  echo "▶ dataset=$dataset shot=$shot sem_kg=$sem vis_kg=$vis seed=$seed"

  python train.py \
    --config configs/default.yaml \
    --dataset "$dataset" \
    --n_shot "$shot" \
    --use_semantic_kg "$sem" \
    --use_visual_kg "$vis" \
    --seed "$seed" \
    --log_dir "./logs/${dataset}_${shot}shot_${tag}_seed${seed}" \
    --checkpoint_dir "./checkpoints/${dataset}_${shot}shot_${tag}_seed${seed}"
}

for dataset in "${DATASETS[@]}"; do
  for shot in "${SHOTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # Baseline: no KG
      run_exp "$dataset" "$shot" "False" "False" "$seed"
      # Semantic KG only
      run_exp "$dataset" "$shot" "True" "False" "$seed"
      # Visual KG only
      run_exp "$dataset" "$shot" "False" "True" "$seed"
      # Full model: both KGs
      run_exp "$dataset" "$shot" "True" "True" "$seed"
    done
  done
done

echo ""
echo "════════════════════════════════════════════════"
echo "All experiments complete!"
echo "Collecting results..."

python - <<'EOF'
import os, json, re
from pathlib import Path

results = {}
for log_dir in Path("./logs").glob("*"):
    log_files = list(log_dir.glob("*.log"))
    if not log_files:
        continue
    tag = log_dir.name
    with open(log_files[-1]) as f:
        lines = f.readlines()
    # Find best val accuracy
    best_acc = 0.0
    for line in lines:
        m = re.search(r"best=(\d+\.\d+)%", line)
        if m:
            best_acc = max(best_acc, float(m.group(1)))
    results[tag] = best_acc

print("\nResults summary:")
print(f"{'Experiment':<60} {'Best Val Acc':>12}")
print("-" * 75)
for tag, acc in sorted(results.items()):
    print(f"{tag:<60} {acc:>11.2f}%")
EOF
