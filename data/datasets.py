"""
data/datasets.py
Dataset loaders for remote sensing few-shot classification.
Supports: UC Merced (UCM), AID, RESISC45, PatternNet.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── Class lists ─────────────────────────────────────────────────────────────

UCM_CLASSES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt",
]

AID_CLASSES = [
    "airport", "bareland", "baseballfield", "beach", "bridge", "center",
    "church", "commercial", "denseresidential", "desert", "farmland",
    "forest", "industrial", "meadow", "mediumresidential", "mountain",
    "park", "parking", "playground", "pond", "port", "railwaystation",
    "resort", "river", "school", "sparseresidential", "square", "stadium",
    "storagetanks", "viaduct",
]

RESISC45_CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
    "bridge", "chaparral", "church", "circular_farmland", "cloud",
    "commercial_area", "dense_residential", "desert", "forest", "freeway",
    "golf_course", "ground_track_field", "harbor", "industrial_area",
    "intersection", "island", "lake", "meadow", "medium_residential",
    "mobile_home_park", "mountain", "overpass", "palace", "parking_lot",
    "railway", "railway_station", "rectangular_farmland", "river",
    "roundabout", "runway", "sea_ice", "ship", "snowberg",
    "sparse_residential", "stadium", "storage_tank", "tennis_court",
    "terrace", "thermal_power_station", "wetland",
]

PATTERNNET_CLASSES = [
    "airplane", "baseball_field", "basketball_court", "beach", "bridge",
    "cemetery", "chaparral", "christmas_tree_farm", "closed_road",
    "coastal_mansion", "crosswalk", "dense_residential", "ferry_terminal",
    "football_field", "forest", "freeway", "golf_course", "harbor",
    "intersection", "mobile_home_park", "nursing_home", "oil_gas_field",
    "oil_well", "overpass", "parking_lot", "parking_space", "railway",
    "river", "runway", "runway_marking", "shipping_yard", "solar_panel",
    "sparse_residential", "storage_tank", "swimming_pool", "tennis_court",
    "transformer_station", "wastewater_treatment_plant",
]

DATASET_INFO = {
    "ucm": {
        "classes": UCM_CLASSES,
        "n_per_class": 100,
        "img_size": 256,
        "url": "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip",
    },
    "aid": {
        "classes": AID_CLASSES,
        "n_per_class": None,  # variable
        "img_size": 600,
        "url": "https://captain-whu.github.io/DiRS/",  # manual download
    },
    "resisc45": {
        "classes": RESISC45_CLASSES,
        "n_per_class": 700,
        "img_size": 256,
        "url": "https://huggingface.co/datasets/jonathan-roberts1/RESISC45",
    },
    "patternnet": {
        "classes": PATTERNNET_CLASSES,
        "n_per_class": 800,
        "img_size": 256,
        "url": "https://sites.google.com/view/zhouwx/dataset",
    },
}


# ── Image transforms ─────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Return train or val/test transforms."""
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP stats
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


# ── Base dataset ─────────────────────────────────────────────────────────────

class RemoteSensingDataset(Dataset):
    """
    Generic ImageFolder-style dataset for remote sensing data.
    Expects: data_root/<dataset_name>/<class_name>/<img>.jpg
    """

    def __init__(
        self,
        root: str,
        dataset: str,
        split: str = "train",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root) / dataset
        self.dataset = dataset.lower()
        self.split = split
        self.transform = transform or get_transforms(split, image_size)

        info = DATASET_INFO[self.dataset]
        self.classes = info["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                # Try with underscores/spaces swapped
                cls_dir = self.root / cls.replace("_", " ")
            if not cls_dir.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif"]:
                for img_path in cls_dir.glob(ext):
                    samples.append((img_path, self.class_to_idx[cls]))
        if not samples:
            raise FileNotFoundError(
                f"No images found in {self.root}. "
                f"Run: bash scripts/download_datasets.sh"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_names(self) -> List[str]:
        return self.classes


# ── Few-shot episode sampler ──────────────────────────────────────────────────

class FewShotEpisodeSampler:
    """
    Samples N-way K-shot episodes.
    Returns (support_images, support_labels, query_images, query_labels).
    """

    def __init__(
        self,
        dataset: RemoteSensingDataset,
        n_way: int = None,        # None = all classes
        n_shot: int = 16,
        n_query: int = 4,
        n_episodes: int = 600,
        seed: int = 42,
    ):
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_episodes = n_episodes
        rng = random.Random(seed)

        # Group samples by class
        self.class_samples: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(dataset.samples):
            self.class_samples.setdefault(label, []).append(idx)

        all_classes = list(self.class_samples.keys())
        self.n_way = n_way or len(all_classes)
        self.all_classes = all_classes

        # Pre-generate episodes
        self.episodes = []
        for _ in range(n_episodes):
            selected = rng.sample(all_classes, self.n_way)
            ep = []
            for cls in selected:
                pool = self.class_samples[cls]
                chosen = rng.sample(pool, min(n_shot + n_query, len(pool)))
                ep.append((cls, chosen[:n_shot], chosen[n_shot:n_shot + n_query]))
            self.episodes.append(ep)

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int):
        return self.episodes[idx]


# ── Helper: build dataloaders ─────────────────────────────────────────────────

def build_dataloader(
    data_root: str,
    dataset: str,
    split: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    n_shot: int = 16,
) -> DataLoader:
    ds = RemoteSensingDataset(data_root, dataset, split, image_size)
    shuffle = (split == "train")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


def build_few_shot_splits(
    data_root: str,
    dataset: str,
    n_shot: int = 16,
    n_query: int = 4,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    image_size: int = 224,
    seed: int = 42,
) -> Tuple[RemoteSensingDataset, RemoteSensingDataset, RemoteSensingDataset]:
    """
    Split dataset into train/val/test (by image indices, not classes).
    For genuine generalisation, consider a class-split strategy instead.
    """
    full = RemoteSensingDataset(data_root, dataset, "train", image_size)
    n = len(full)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    train_idx = indices[n_val + n_test:]
    val_idx = indices[n_test:n_val + n_test]
    test_idx = indices[:n_test]

    def make_subset(idx_list, split):
        ds = RemoteSensingDataset(data_root, dataset, split, image_size)
        ds.samples = [full.samples[i] for i in idx_list]
        return ds

    return (
        make_subset(train_idx, "train"),
        make_subset(val_idx, "val"),
        make_subset(test_idx, "test"),
    )
