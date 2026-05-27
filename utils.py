"""
utils.py
Utilities: seeding, logging, checkpointing, metrics, config loading.
"""

import os
import random
import logging
import time
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch


# ── Seeding ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fully deterministic seeding."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            d = cfg
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
    return cfg


def cfg_to_str(cfg: Dict) -> str:
    """Pretty-print config for logging."""
    lines = ["── Config ──"]
    for k, v in sorted(cfg.items()):
        if isinstance(v, dict):
            lines.append(f"  {k}:")
            for kk, vv in v.items():
                lines.append(f"    {kk}: {vv}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_dir: str, level=logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{ts}.log"

    fmt = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class AverageMeter:
    """Tracks running average for a metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricTracker:
    """Tracks multiple metrics over training."""

    def __init__(self):
        self.history: Dict[str, list] = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(v)

    def latest(self, key: str) -> float:
        return self.history[key][-1] if key in self.history else 0.0

    def best(self, key: str, mode: str = "max") -> float:
        vals = self.history.get(key, [0.0])
        return max(vals) if mode == "max" else min(vals)

    def summary(self) -> str:
        lines = []
        for k, v in self.history.items():
            lines.append(f"  {k}: best={max(v):.4f}, last={v[-1]:.4f}")
        return "\n".join(lines)


# ── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / filename
    torch.save(state, path)
    if is_best:
        best_path = Path(checkpoint_dir) / "best_model.pth"
        shutil.copyfile(path, best_path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)
    best_acc = ckpt.get("best_acc", 0.0)
    return epoch, best_acc


# ── Metrics ──────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        results = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results[f"top{k}"] = correct_k.mul_(100.0 / batch_size).item()
        return results


def per_class_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
) -> Dict[int, float]:
    """Compute per-class accuracy."""
    preds = logits.argmax(dim=1)
    acc = {}
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() == 0:
            continue
        acc[c] = (preds[mask] == targets[mask]).float().mean().item() * 100
    return acc


def confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        cm[t.long(), p.long()] += 1
    return cm


# ── Device helpers ───────────────────────────────────────────────────────────

def get_device(device_str: str = "cuda") -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ── Timer ────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> str:
        s = time.time() - self._start
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def reset(self):
        self._start = time.time()
