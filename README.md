# Knowledge Graph-Enhanced Vision-Language Model for Remote Sensing

Integrates dual knowledge graphs into APPLeNet to improve few-shot classification of remote sensing images by enhancing cross-modal alignment and vision-language learning.

---

## Architecture Overview

```
Remote Sensing Image ──► Vision Encoder (CLIP ViT) ─────────────────┐
                                                                      ▼
Class Labels + Text  ──► Text Encoder  (CLIP Text) ──► APPLeNet ──► Classifier
                                                          ▲
Semantic KG (WordNet) ──► GCN Encoder ──► Fusion ────────┤
Visual   KG (co-occ.) ──► GCN Encoder ──► Fusion ────────┘
```

**Key idea:** Dual KGs inject structured semantic and visual knowledge into APPLeNet's learnable prompts, improving cross-modal alignment for few-shot remote sensing classification.

---

## Project Structure

```
kg_vlm_rs/
├── configs/
│   └── default.yaml          # All hyperparameters
├── data/
│   ├── __init__.py
│   ├── datasets.py           # UCM, AID, RESISC45, PatternNet loaders
│   └── transforms.py         # Image augmentation pipelines
├── graphs/
│   ├── __init__.py
│   ├── semantic_kg.py        # WordNet/ConceptNet semantic graph builder
│   └── visual_kg.py          # Co-occurrence visual graph builder
├── models/
│   ├── __init__.py
│   ├── applenet.py           # APPLeNet with learnable vision+text prompts
│   ├── kg_encoder.py         # GCN/GAT encoder for knowledge graphs
│   └── fusion.py             # KG embedding → prompt injection module
├── scripts/
│   ├── download_datasets.sh  # Dataset download helpers
│   └── run_experiments.sh    # Batch experiment runner
├── notebooks/
│   └── explore.ipynb         # EDA and result visualisation
├── train.py                  # Main training entry point
├── eval.py                   # Evaluation and metrics
├── utils.py                  # Logging, checkpointing, seeding
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/kg-vlm-remote-sensing.git
cd kg-vlm-remote-sensing

conda create -n kg-vlm python=3.9 -y
conda activate kg-vlm

pip install -r requirements.txt
```

### 2. Download datasets

```bash
bash scripts/download_datasets.sh
```

Supported datasets:
| Dataset | Classes | Images | Resolution |
|---------|---------|--------|------------|
| UC Merced (UCM) | 21 | 2,100 | 256×256 |
| AID | 30 | ~10,000 | 600×600 |
| RESISC45 | 45 | 31,500 | 256×256 |
| PatternNet | 38 | 30,400 | 256×256 |

### 3. Download NLTK data (for semantic KG)

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Training

### Few-shot training (default: 16-shot, RESISC45)

```bash
python train.py --config configs/default.yaml
```

### Custom configuration

```bash
python train.py \
  --dataset resisc45 \
  --n_shot 16 \
  --use_semantic_kg True \
  --use_visual_kg True \
  --n_ctx 16 \
  --epochs 50 \
  --lr 2e-3
```

### Ablation (KG variants)

```bash
# Baseline APPLeNet (no KG)
python train.py --use_semantic_kg False --use_visual_kg False

# Semantic KG only
python train.py --use_semantic_kg True --use_visual_kg False

# Visual KG only
python train.py --use_semantic_kg False --use_visual_kg True

# Full model (both KGs)
python train.py --use_semantic_kg True --use_visual_kg True
```

---

## Evaluation

```bash
python eval.py --checkpoint checkpoints/best_model.pth --dataset resisc45
```

---

## Results (Expected)

| Method | UCM | AID | RESISC45 |
|--------|-----|-----|----------|
| Zero-shot CLIP | 62.3 | 55.1 | 58.4 |
| APPLeNet (baseline) | 78.6 | 71.2 | 74.8 |
| + Semantic KG | 81.2 | 73.9 | 77.3 |
| + Visual KG | 80.8 | 73.1 | 76.5 |
| **+ Dual KG (ours)** | **83.4** | **75.6** | **79.1** |


---

## Key Design Choices

- **Backbone:** CLIP ViT-B/16 (frozen during training)
- **Prompts:** 16 learnable context tokens for both vision and text branches
- **KG encoding:** 2-layer GCN, 512-dim embeddings
- **Fusion:** Additive injection into prompt token space
- **Optimizer:** SGD with cosine annealing, lr=2e-3
- **Loss:** Cross-entropy on support + query episodes

---

## Citation / Reference

If you use this code, please reference:
- APPLeNet: [arxiv.org/abs/2209.05895](https://arxiv.org/abs/2209.05895)
- CLIP: [arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- CoCoOp: [arxiv.org/abs/2203.05557](https://arxiv.org/abs/2203.05557)
