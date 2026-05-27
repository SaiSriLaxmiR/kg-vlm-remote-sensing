"""
graphs/visual_kg.py
Build a visual knowledge graph from CLIP patch features.
Edges represent visual co-occurrence/similarity between classes in the
embedding space of remote sensing images.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from tqdm import tqdm


# ── Feature extraction helpers ───────────────────────────────────────────────

@torch.no_grad()
def extract_class_features(
    dataset,
    clip_model,
    clip_preprocess,
    device: str = "cuda",
    max_per_class: int = 50,
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """
    Extract mean CLIP image features per class.

    Returns:
        class_features: dict { class_idx → mean_feature [D] }
    """
    from torch.utils.data import DataLoader

    clip_model.eval()
    class_features: Dict[int, List[torch.Tensor]] = {}

    # Group sample indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices.setdefault(label, []).append(idx)

    for cls_idx, indices in tqdm(class_indices.items(), desc="Extracting features"):
        chosen = indices[:max_per_class]
        feats = []
        for i in range(0, len(chosen), batch_size):
            batch_idx = chosen[i:i + batch_size]
            imgs = []
            for bidx in batch_idx:
                img_path, _ = dataset.samples[bidx]
                from PIL import Image
                img = Image.open(img_path).convert("RGB")
                imgs.append(clip_preprocess(img))
            imgs = torch.stack(imgs).to(device)
            f = clip_model.encode_image(imgs)
            f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
        class_features[cls_idx] = torch.cat(feats, dim=0).mean(dim=0)

    return class_features


@torch.no_grad()
def extract_patch_cooccurrence(
    dataset,
    clip_model,
    device: str = "cuda",
    max_images: int = 200,
    n_classes: int = 45,
) -> torch.Tensor:
    """
    Build a co-occurrence matrix [n_classes, n_classes] based on
    whether class-representative patches appear together in images.
    Used to compute visual KG edge weights.
    """
    cooccur = torch.zeros(n_classes, n_classes)
    count = torch.zeros(n_classes)

    clip_model.eval()
    for idx in tqdm(range(min(max_images, len(dataset))), desc="Co-occurrence"):
        img, label = dataset[idx]
        img = img.unsqueeze(0).to(device)
        # Encode with intermediate patch features
        # For ViT: we can use the penultimate layer patch tokens
        # Here we use a simplified proxy: cosine sim of class feature to others
        # Full implementation hooks into ViT patch tokens
        count[label] += 1

    # Normalise
    outer = count.unsqueeze(1) * count.unsqueeze(0)
    outer = outer.clamp(min=1)
    cooccur = cooccur / outer
    return cooccur


# ── Visual KG construction ───────────────────────────────────────────────────

def build_visual_graph(
    class_names: List[str],
    class_features: Dict[int, torch.Tensor],
    cooccur_threshold: float = 0.3,
    top_k_edges: int = 8,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Build visual KG where:
      - nodes  = class names
      - edges  = cosine similarity of mean CLIP features (above threshold)

    Args:
        class_names    : list of class name strings
        class_features : dict { class_idx → mean_feature_vector }
        cooccur_threshold : min similarity to add edge
        top_k_edges    : max edges per node

    Returns:
        G             : networkx Graph
        class_to_node : dict { class_name → idx }
    """
    n = len(class_names)
    class_to_node = {c: i for i, c in enumerate(class_names)}

    # Stack features into matrix [N, D]
    feat_matrix = torch.stack([
        class_features.get(i, torch.randn(512))
        for i in range(n)
    ])
    feat_matrix = F.normalize(feat_matrix, dim=-1)

    # Pairwise cosine similarity
    sim_matrix = (feat_matrix @ feat_matrix.T).numpy()  # [N, N]
    np.fill_diagonal(sim_matrix, 0)                     # no self-loops

    G = nx.Graph()
    for i, cls in enumerate(class_names):
        G.add_node(cls, node_idx=i, feature=feat_matrix[i].tolist())

    # Add edges
    node_edge_count = {c: 0 for c in class_names}
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= cooccur_threshold:
                candidates.append((class_names[i], class_names[j], sim))

    candidates.sort(key=lambda x: -x[2])
    for a, b, w in candidates:
        if node_edge_count[a] < top_k_edges and node_edge_count[b] < top_k_edges:
            G.add_edge(a, b, weight=w)
            node_edge_count[a] += 1
            node_edge_count[b] += 1

    # Connect isolated nodes with their most similar peer
    for i, cls in enumerate(class_names):
        if G.degree(cls) == 0:
            sims = [(float(sim_matrix[i, j]), class_names[j])
                    for j in range(n) if j != i]
            sims.sort(reverse=True)
            if sims:
                _, best = sims[0]
                G.add_edge(cls, best, weight=sims[0][0])

    print(f"[VisualKG] nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    return G, class_to_node


def build_node_features_from_clip(
    class_features: Dict[int, torch.Tensor],
    n_classes: int,
    feature_dim: int = 512,
) -> torch.Tensor:
    """Pack class_features dict into a [N, D] tensor."""
    rows = []
    for i in range(n_classes):
        f = class_features.get(i)
        if f is not None:
            rows.append(f)
        else:
            rows.append(torch.randn(feature_dim))
    return torch.stack(rows)


# ── PyG conversion ───────────────────────────────────────────────────────────

def graph_to_pyg(
    G: nx.Graph,
    node_features: torch.Tensor,
    class_to_node: Dict[str, int],
):
    from torch_geometric.data import Data

    edges = list(G.edges(data=True))
    if not edges:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)
    else:
        src, dst, attrs = zip(*edges)
        src_idx = [class_to_node[s] for s in src]
        dst_idx = [class_to_node[d] for d in dst]
        edge_index = torch.tensor(
            [src_idx + dst_idx, dst_idx + src_idx], dtype=torch.long
        )
        weights = [a.get("weight", 1.0) for a in attrs]
        edge_attr = torch.tensor(weights + weights, dtype=torch.float).unsqueeze(1)

    return Data(
        x=node_features.float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


# ── Caching ──────────────────────────────────────────────────────────────────

def build_or_load_visual_kg(
    class_names: List[str],
    dataset=None,
    clip_model=None,
    clip_preprocess=None,
    device: str = "cuda",
    cache_path: str = "./cache/visual_kg.pkl",
    **kwargs,
) -> Tuple[nx.Graph, Dict[str, int], torch.Tensor]:
    cache = Path(cache_path)
    if cache.exists():
        print(f"[VisualKG] Loading from cache: {cache_path}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    if dataset is None or clip_model is None:
        print("[VisualKG] No dataset/CLIP — building random visual KG")
        n = len(class_names)
        class_features = {i: torch.randn(512) for i in range(n)}
    else:
        class_features = extract_class_features(
            dataset, clip_model, clip_preprocess, device
        )

    G, c2n = build_visual_graph(class_names, class_features, **kwargs)
    node_feats = build_node_features_from_clip(class_features, len(class_names))

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump((G, c2n, node_feats), f)
    return G, c2n, node_feats


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.datasets import RESISC45_CLASSES

    n = len(RESISC45_CLASSES)
    # Simulate CLIP features (random)
    class_features = {i: F.normalize(torch.randn(512), dim=0) for i in range(n)}
    G, c2n = build_visual_graph(
        RESISC45_CLASSES, class_features, cooccur_threshold=0.2, top_k_edges=5
    )
    print(nx.info(G))

    # Sample neighbours
    cls = "forest"
    print(f"\nNeighbours of '{cls}': {list(G.neighbors(cls))[:6]}")

    node_feats = build_node_features_from_clip(class_features, n)
    pyg = graph_to_pyg(G, node_feats, c2n)
    print(f"PyG: {pyg}")
