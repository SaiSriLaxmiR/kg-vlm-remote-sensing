"""
train.py
Main training entry point for KG-APPLeNet.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import (
    set_seed, load_config, cfg_to_str, setup_logger,
    AverageMeter, MetricTracker, save_checkpoint,
    accuracy, get_device, count_params, Timer,
)
from data.datasets import (
    RemoteSensingDataset, DATASET_INFO,
    build_few_shot_splits, build_dataloader, get_transforms,
)
from graphs.semantic_kg import build_or_load_semantic_kg, build_node_features, graph_to_pyg
from graphs.visual_kg import build_or_load_visual_kg, build_node_features_from_clip, graph_to_pyg as vis_graph_to_pyg
from models.applenet import build_model


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="KG-APPLeNet Training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--n_shot", type=int)
    parser.add_argument("--use_semantic_kg", type=lambda x: x.lower() == "true")
    parser.add_argument("--use_visual_kg", type=lambda x: x.lower() == "true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    return parser.parse_args()


# ── KG preparation ───────────────────────────────────────────────────────────

def prepare_kg_data(cfg, class_names, clip_model, device, logger, train_ds=None):
    """Build/load both KGs and return PyG Data objects on device."""
    os.makedirs("./cache", exist_ok=True)
    dataset_tag = cfg["dataset"]

    sem_data, vis_data = None, None

    if cfg.get("use_semantic_kg", True):
        logger.info("Building semantic KG...")
        G_sem, c2n_sem = build_or_load_semantic_kg(
            class_names,
            cache_path=f"./cache/sem_kg_{dataset_tag}.pkl",
            similarity_threshold=cfg.get("semantic_kg", {}).get("similarity_threshold", 0.25),
            top_k_edges=cfg.get("semantic_kg", {}).get("top_k_edges", 10),
        )
        sem_feats = build_node_features(class_names, clip_model, None, device)
        sem_data = graph_to_pyg(G_sem, sem_feats, c2n_sem).to(device)
        logger.info(f"Semantic KG: {G_sem.number_of_nodes()} nodes, {G_sem.number_of_edges()} edges")

    if cfg.get("use_visual_kg", True):
        logger.info("Building visual KG...")
        result = build_or_load_visual_kg(
            class_names,
            dataset=train_ds,
            clip_model=clip_model if train_ds else None,
            cache_path=f"./cache/vis_kg_{dataset_tag}.pkl",
            cooccur_threshold=cfg.get("visual_kg", {}).get("cooccur_threshold", 0.2),
            top_k_edges=cfg.get("visual_kg", {}).get("top_k_edges", 8),
        )
        G_vis, c2n_vis, vis_feats = result
        vis_data = vis_graph_to_pyg(G_vis, vis_feats, c2n_vis).to(device)
        logger.info(f"Visual KG: {G_vis.number_of_nodes()} nodes, {G_vis.number_of_edges()} edges")

    return sem_data, vis_data


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, sem_data, vis_data, logger, epoch):
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc@1")

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(images, sem_data, vis_data)
        logits = out["logits"]

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accs = accuracy(logits.detach(), labels)
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(accs["top1"], images.size(0))

        if (step + 1) % 10 == 0:
            logger.info(
                f"  Epoch {epoch} [{step+1}/{len(loader)}] "
                f"loss={loss_meter.avg:.4f} acc={acc_meter.avg:.2f}%"
            )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, sem_data, vis_data):
    model.eval()
    loss_meter = AverageMeter("loss")
    acc1_meter = AverageMeter("acc@1")
    acc5_meter = AverageMeter("acc@5")

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(images, sem_data, vis_data)
        logits = out["logits"]
        loss = criterion(logits, labels)

        n_cls = logits.shape[1]
        topk = (1, min(5, n_cls))
        accs = accuracy(logits, labels, topk=topk)

        loss_meter.update(loss.item(), images.size(0))
        acc1_meter.update(accs["top1"], images.size(0))
        if "top5" in accs:
            acc5_meter.update(accs["top5"], images.size(0))

    return {
        "loss": loss_meter.avg,
        "acc1": acc1_meter.avg,
        "acc5": acc5_meter.avg,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config + apply CLI overrides
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("config", "resume")}
    cfg = load_config(args.config, overrides)

    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))

    log_dir = cfg.get("log_dir", "./logs")
    logger = setup_logger("train", log_dir)
    logger.info(cfg_to_str(cfg))
    logger.info(f"Device: {device}")

    # Dataset info
    dataset = cfg["dataset"]
    class_names = DATASET_INFO[dataset]["classes"]
    n_classes = len(class_names)
    logger.info(f"Dataset: {dataset} | {n_classes} classes | {cfg['n_shot']}-shot")

    # Build model
    logger.info("Building model...")
    model, clip_preprocess = build_model(cfg, class_names, str(device))

    # Dataset splits
    train_ds, val_ds, test_ds = build_few_shot_splits(
        data_root=cfg.get("data_root", "./datasets"),
        dataset=dataset,
        n_shot=cfg.get("n_shot", 16),
        n_query=cfg.get("n_query", 4),
        image_size=cfg.get("image_size", 224),
        seed=cfg.get("seed", 42),
    )

    train_loader = build_dataloader(
        cfg["data_root"], dataset, "train",
        cfg["image_size"], cfg["batch_size"], cfg["num_workers"]
    )
    val_loader = build_dataloader(
        cfg["data_root"], dataset, "val",
        cfg["image_size"], cfg["batch_size"], cfg["num_workers"]
    )

    # KG data
    sem_data, vis_data = prepare_kg_data(
        cfg, class_names, model.clip_model, device, logger, train_ds
    )

    # Optimizer (only trainable params)
    trainable = model.get_trainable_params()
    param_counts = count_params(model)
    logger.info(
        f"Parameters: total={param_counts['total']:,} "
        f"trainable={param_counts['trainable']:,} "
        f"frozen={param_counts['frozen']:,}"
    )

    opt_name = cfg.get("optimizer", "sgd").lower()
    if opt_name == "adamw":
        optimizer = AdamW(trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = SGD(
            trainable,
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg["weight_decay"],
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and Path(args.resume).exists():
        from utils import load_checkpoint
        start_epoch, best_acc = load_checkpoint(args.resume, model, optimizer, str(device))
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch}, best_acc={best_acc:.2f})")

    tracker = MetricTracker()
    timer = Timer()

    # Training
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(start_epoch + 1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, sem_data, vis_data, logger, epoch
        )
        scheduler.step()

        if epoch % cfg.get("eval_interval", 5) == 0:
            val_metrics = evaluate(model, val_loader, criterion, device, sem_data, vis_data)
            is_best = val_metrics["acc1"] > best_acc
            if is_best:
                best_acc = val_metrics["acc1"]

            tracker.update({
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_acc": val_metrics["acc1"],
            })

            logger.info(
                f"Epoch {epoch}/{cfg['epochs']} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['acc']:.2f}% "
                f"val_acc={val_metrics['acc1']:.2f}% "
                f"best={best_acc:.2f}% | "
                f"elapsed={timer.elapsed()}"
            )

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "cfg": cfg,
                },
                cfg.get("checkpoint_dir", "./checkpoints"),
                filename=f"ckpt_epoch{epoch}.pth",
                is_best=is_best,
            )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best val accuracy: {best_acc:.2f}%")
    logger.info(f"Total time: {timer.elapsed()}")
    logger.info(tracker.summary())


if __name__ == "__main__":
    main()
