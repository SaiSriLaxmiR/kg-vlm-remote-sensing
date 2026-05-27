"""
graphs/semantic_kg.py
Build a semantic knowledge graph from WordNet for remote sensing class names.
Nodes = classes. Edges = shared hypernyms, siblings, or WordNet similarity.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import networkx as nx

try:
    from nltk.corpus import wordnet as wn
    import nltk
    NLTK_OK = True
except ImportError:
    NLTK_OK = False


def _ensure_wordnet():
    if not NLTK_OK:
        raise ImportError("Install nltk: pip install nltk")
    try:
        wn.synsets("forest")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


# ── WordNet utilities ────────────────────────────────────────────────────────

def get_synset(word: str):
    """Return the most relevant synset for a class name."""
    _ensure_wordnet()
    # Clean compound class names
    word_clean = word.replace("_", " ").lower()
    candidates = wn.synsets(word_clean.replace(" ", "_"), pos=wn.NOUN)
    if not candidates:
        # Try first token
        first = word_clean.split()[0]
        candidates = wn.synsets(first, pos=wn.NOUN)
    return candidates[0] if candidates else None


def wup_similarity(word_a: str, word_b: str) -> float:
    """Wu-Palmer similarity between two class names."""
    s1 = get_synset(word_a)
    s2 = get_synset(word_b)
    if s1 is None or s2 is None:
        return 0.0
    sim = s1.wup_similarity(s2)
    return sim if sim is not None else 0.0


def path_similarity(word_a: str, word_b: str) -> float:
    s1 = get_synset(word_a)
    s2 = get_synset(word_b)
    if s1 is None or s2 is None:
        return 0.0
    sim = s1.path_similarity(s2)
    return sim if sim is not None else 0.0


def get_definition(word: str) -> str:
    """Return WordNet definition string for embedding."""
    syn = get_synset(word)
    if syn:
        return syn.definition()
    return word.replace("_", " ")


def get_hypernym_path(word: str) -> List[str]:
    """Return list of hypernyms (generalisation chain)."""
    syn = get_synset(word)
    if syn is None:
        return []
    path = syn.hypernym_paths()
    if not path:
        return []
    return [s.name().split(".")[0] for s in path[0]]


# ── Graph construction ───────────────────────────────────────────────────────

def build_semantic_graph(
    class_names: List[str],
    similarity_threshold: float = 0.25,
    top_k_edges: int = 10,
    use_path_sim: bool = False,
) -> Tuple[nx.Graph, Dict[str, int]]:
    """
    Build semantic KG where:
      - nodes  = class names
      - edges  = weighted by WuP / path similarity (above threshold)
    
    Returns:
        G              : networkx Graph with 'weight' on edges
        class_to_node  : dict mapping class name → node index
    """
    _ensure_wordnet()
    G = nx.Graph()
    n = len(class_names)
    class_to_node = {c: i for i, c in enumerate(class_names)}

    # Add nodes with WordNet metadata
    for cls in class_names:
        defn = get_definition(cls)
        hypernyms = get_hypernym_path(cls)
        G.add_node(
            cls,
            definition=defn,
            hypernyms=hypernyms,
            node_idx=class_to_node[cls],
        )

    # Add edges based on similarity
    sim_fn = path_similarity if use_path_sim else wup_similarity
    edge_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = class_names[i], class_names[j]
            score = sim_fn(a, b)
            if score >= similarity_threshold:
                edge_scores.append((a, b, score))

    # Keep only top-K edges per node
    node_edge_count = {c: 0 for c in class_names}
    edge_scores.sort(key=lambda x: -x[2])
    for a, b, w in edge_scores:
        if node_edge_count[a] < top_k_edges and node_edge_count[b] < top_k_edges:
            G.add_edge(a, b, weight=w)
            node_edge_count[a] += 1
            node_edge_count[b] += 1

    # Ensure connectivity: add fallback weak edges for isolated nodes
    for cls in class_names:
        if G.degree(cls) == 0:
            scores = [(wup_similarity(cls, other), other)
                      for other in class_names if other != cls]
            scores.sort(reverse=True)
            if scores:
                _, best = scores[0]
                G.add_edge(cls, best, weight=0.1)

    print(f"[SemanticKG] nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    return G, class_to_node


# ── Node feature extraction ──────────────────────────────────────────────────

def build_node_features(
    class_names: List[str],
    clip_model=None,
    clip_tokenizer=None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build node feature matrix using CLIP text encoder on class definitions.
    Falls back to random features if CLIP is not provided.

    Returns:
        Tensor of shape [n_classes, feature_dim]
    """
    if clip_model is not None and clip_tokenizer is not None:
        import open_clip
        clip_model.eval()
        features = []
        with torch.no_grad():
            for cls in class_names:
                defn = get_definition(cls)
                prompt = f"a satellite image of {cls.replace('_', ' ')}: {defn}"
                tokens = clip_tokenizer([prompt]).to(device)
                feat = clip_model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                features.append(feat.squeeze(0).cpu())
        return torch.stack(features)  # [N, D]
    else:
        # Random init (to be trained)
        print("[SemanticKG] CLIP not available — using random node features")
        return torch.randn(len(class_names), 512)


# ── PyG conversion ───────────────────────────────────────────────────────────

def graph_to_pyg(
    G: nx.Graph,
    node_features: torch.Tensor,
    class_to_node: Dict[str, int],
):
    """
    Convert networkx Graph → PyTorch Geometric Data object.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("Install torch-geometric: pip install torch-geometric")

    # Build edge_index and edge_attr
    edges = list(G.edges(data=True))
    if not edges:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 1)
    else:
        src, dst, attrs = zip(*edges)
        src_idx = [class_to_node[s] for s in src]
        dst_idx = [class_to_node[d] for d in dst]
        # Undirected: add both directions
        edge_index = torch.tensor(
            [src_idx + dst_idx, dst_idx + src_idx], dtype=torch.long
        )
        weights = [a.get("weight", 1.0) for a in attrs]
        edge_attr = torch.tensor(weights + weights, dtype=torch.float).unsqueeze(1)

    data = Data(
        x=node_features.float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data


# ── Caching ──────────────────────────────────────────────────────────────────

def build_or_load_semantic_kg(
    class_names: List[str],
    cache_path: str = "./cache/semantic_kg.pkl",
    **kwargs,
) -> Tuple[nx.Graph, Dict[str, int]]:
    cache = Path(cache_path)
    if cache.exists():
        print(f"[SemanticKG] Loading from cache: {cache_path}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("[SemanticKG] Building from scratch...")
    G, c2n = build_semantic_graph(class_names, **kwargs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump((G, c2n), f)
    return G, c2n


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data.datasets import RESISC45_CLASSES

    G, c2n = build_semantic_graph(
        RESISC45_CLASSES,
        similarity_threshold=0.3,
        top_k_edges=5,
    )
    print(nx.info(G))

    # Sample neighbours
    test_cls = "forest"
    nbrs = list(G.neighbors(test_cls))
    print(f"\nNeighbours of '{test_cls}': {nbrs[:8]}")

    # Node features (random since no CLIP)
    feats = build_node_features(RESISC45_CLASSES)
    print(f"Node feature shape: {feats.shape}")

    # PyG
    pyg_data = graph_to_pyg(G, feats, c2n)
    print(f"PyG Data: {pyg_data}")
