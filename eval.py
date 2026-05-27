"""
eval.py
Evaluation script: loads checkpoint, runs on test set, reports metrics.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from utils import (
    set_seed, load_config, setup_logger, get_device,
    accuracy, per_class_accuracy, confusion_matrix,
)
from data.datasets import DATASET_INFO, build_dataloader
from models.applenet import build_model
from graphs.semantic_kg import build_or_load_semantic_kg, build_node_features, graph_to_pyg
from graphs.visual_kg import build_or_load_visual_kg, graph_to_pyg as vis_graph_to_pyg

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dataset", type=str)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_shot", type=int)
    return p.parse_args()


@torch.no_grad()
def run_eval(model, loader, device, sem_data, vis_data, class_names):
    model.eval()
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images, sem_data, vis_data)
        all_logits.append(out["logits"].cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    n_cls = len(class_names)
    topk = (1, min(5, n_cls))
    accs = accuracy(logits, labels, topk=topk)
    per_cls = per_class_accuracy(logits, labels, n_cls)
    cm = confusion_matrix(logits, labels, n_cls)

    return {
        "top1": accs["top1"],
        "top5": accs.get("top5", 0.0),
        "per_class": per_cls,
        "confusion_matrix": cm,
        "logits": logits,
        "labels": labels,
    }


def print_results(results, class_names, logger):
    logger.info("=" * 60)
    logger.info(f"Top-1 Accuracy: {results['top1']:.2f}%")
    logger.info(f"Top-5 Accuracy: {results['top5']:.2f}%")
    logger.info("─" * 60)
    logger.info("Per-class accuracy:")
    per_cls = results["per_class"]
    worst = sorted(per_cls.items(), key=lambda x: x[1])[:5]
    best = sorted(per_cls.items(), key=lambda x: -x[1])[:5]
    logger.info("  Best 5 classes:")
    for idx, acc in best:
        logger.info(f"    {class_names[idx]:<35} {acc:.1f}%")
    logger.info("  Worst 5 classes:")
    for idx, acc in worst:
        logger.info(f"    {class_names[idx]:<35} {acc:.1f}%")
    logger.info("=" * 60)


def save_confusion_matrix(cm, class_names, out_path="./logs/confusion_matrix.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(14, 12))
        cm_norm = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1)
        im = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        tick = range(len(class_names))
        ax.set_xticks(tick)
        ax.set_yticks(tick)
        labels_short = [c[:12] for c in class_names]
        ax.set_xticklabels(labels_short, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_short, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalised Confusion Matrix")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        print(f"Confusion matrix saved to {out_path}")
    except Exception as e:
        print(f"Could not save confusion matrix: {e}")


def main():
    args = parse_args()
    device = get_device(args.device)
    logger = setup_logger("eval", "./logs")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=str(device))
    cfg = ckpt.get("cfg", {})

    # CLI overrides
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.n_shot:
        cfg["n_shot"] = args.n_shot

    set_seed(cfg.get("seed", 42))
    dataset = cfg["dataset"]
    class_names = DATASET_INFO[dataset]["classes"]

    logger.info(f"Evaluating on {dataset} with {len(class_names)} classes")

    # Model
    model, _ = build_model(cfg, class_names, str(device))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Test loader
    test_loader = build_dataloader(
        cfg.get("data_root", "./datasets"),
        dataset, "test",
        cfg.get("image_size", 224),
        batch_size=64,
        num_workers=4,
    )

    # KG data
    sem_data, vis_data = None, None
    if cfg.get("use_semantic_kg", True):
        G_sem, c2n_sem = build_or_load_semantic_kg(
            class_names, cache_path=f"./cache/sem_kg_{dataset}.pkl"
        )
        sem_feats = build_node_features(class_names)
        sem_data = graph_to_pyg(G_sem, sem_feats, c2n_sem).to(device)

    if cfg.get("use_visual_kg", True):
        result = build_or_load_visual_kg(
            class_names, cache_path=f"./cache/vis_kg_{dataset}.pkl"
        )
        G_vis, c2n_vis, vis_feats = result
        vis_data = vis_graph_to_pyg(G_vis, vis_feats, c2n_vis).to(device)

    # Run evaluation
    results = run_eval(model, test_loader, device, sem_data, vis_data, class_names)
    print_results(results, class_names, logger)
    save_confusion_matrix(results["confusion_matrix"], class_names)


if __name__ == "__main__":
    main()
