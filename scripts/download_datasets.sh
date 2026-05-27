#!/usr/bin/env bash
# scripts/download_datasets.sh
# Download remote sensing datasets for KG-APPLeNet.

set -e

DATA_ROOT="${1:-./datasets}"
mkdir -p "$DATA_ROOT"

echo "════════════════════════════════════════"
echo "  Remote Sensing Dataset Downloader"
echo "  Saving to: $DATA_ROOT"
echo "════════════════════════════════════════"

# ── UC Merced Land Use (UCM) ─────────────────────────────────────────────────
download_ucm() {
  local out="$DATA_ROOT/ucm"
  if [ -d "$out" ]; then
    echo "[UCM] Already downloaded, skipping."
    return
  fi
  echo "[UCM] Downloading UC Merced Land Use dataset..."
  local url="http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"
  wget -q --show-progress -O /tmp/ucm.zip "$url"
  unzip -q /tmp/ucm.zip -d /tmp/
  mv "/tmp/UCMerced_LandUse/Images" "$out"
  rm -rf /tmp/ucm.zip /tmp/UCMerced_LandUse
  echo "[UCM] Done. Classes: $(ls $out | wc -l)"
}

# ── RESISC45 (via HuggingFace) ───────────────────────────────────────────────
download_resisc45() {
  local out="$DATA_ROOT/resisc45"
  if [ -d "$out" ]; then
    echo "[RESISC45] Already downloaded, skipping."
    return
  fi
  echo "[RESISC45] Downloading RESISC45..."
  python3 - <<'EOF'
from datasets import load_dataset
import os, shutil
from PIL import Image

ds = load_dataset("jonathan-roberts1/RESISC45", split="train")
out = "./datasets/resisc45"
os.makedirs(out, exist_ok=True)

for item in ds:
    label = item["label"]
    class_name = ds.features["label"].int2str(label)
    cls_dir = os.path.join(out, class_name)
    os.makedirs(cls_dir, exist_ok=True)
    img = item["image"]
    n = len(os.listdir(cls_dir))
    img.save(os.path.join(cls_dir, f"{n:05d}.jpg"))

print(f"RESISC45 done. Classes: {len(os.listdir(out))}")
EOF
}

# ── PatternNet ───────────────────────────────────────────────────────────────
download_patternnet() {
  local out="$DATA_ROOT/patternnet"
  if [ -d "$out" ]; then
    echo "[PatternNet] Already downloaded, skipping."
    return
  fi
  echo "[PatternNet] PatternNet requires manual download."
  echo "  Visit: https://sites.google.com/view/zhouwx/dataset"
  echo "  Download PatternNet.zip, extract to: $DATA_ROOT/patternnet"
}

# ── AID ──────────────────────────────────────────────────────────────────────
download_aid() {
  local out="$DATA_ROOT/aid"
  if [ -d "$out" ]; then
    echo "[AID] Already downloaded, skipping."
    return
  fi
  echo "[AID] AID requires manual download from Google Drive."
  echo "  Visit: https://captain-whu.github.io/DiRS/"
  echo "  Or: https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets"
  echo "  Extract to: $DATA_ROOT/aid"
}

# ── Run downloads ────────────────────────────────────────────────────────────
echo ""
echo "Downloading UCM..."
download_ucm

echo ""
echo "Downloading RESISC45..."
download_resisc45

echo ""
echo "Checking PatternNet..."
download_patternnet

echo ""
echo "Checking AID..."
download_aid

echo ""
echo "════════════════════════════════════════"
echo "Download complete!"
echo "Dataset directory: $DATA_ROOT"
ls "$DATA_ROOT" 2>/dev/null | while read d; do
  n=$(find "$DATA_ROOT/$d" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
  echo "  $d: $n images"
done
echo "════════════════════════════════════════"
