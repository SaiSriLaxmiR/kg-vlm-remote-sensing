"""
models/kg_encoder.py
GCN and GAT encoders for knowledge graph node embeddings.
Encodes both semantic and visual KGs into a shared embedding space
compatible with CLIP's token dimension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    PYG_OK = True
except ImportError:
    PYG_OK = False


# ── GCN Encoder ──────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """
    2-layer Graph Convolutional Network.
    Encodes node features into fixed-dim embeddings.

    Input : x [N, in_dim], edge_index [2, E]
    Output: z [N, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        if not PYG_OK:
            raise ImportError("pip install torch-geometric")

        self.residual = residual
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i + 1]) for i in range(n_layers)
        ])
        self.dropout = dropout

        # Projection for residual if dims change
        self.residual_proj = None
        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if self.residual and h.shape == h_new.shape:
                h_new = h_new + h
            elif self.residual and i == len(self.convs) - 1 and self.residual_proj:
                h_new = h_new + self.residual_proj(x)
            h = h_new
        return h


# ── GAT Encoder ──────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    2-layer Graph Attention Network.
    Uses multi-head attention to weight neighbour contributions.

    Input : x [N, in_dim], edge_index [2, E]
    Output: z [N, out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 512,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not PYG_OK:
            raise ImportError("pip install torch-geometric")

        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv1(h, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h


# ── Dual KG Encoder ──────────────────────────────────────────────────────────

class DualKGEncoder(nn.Module):
    """
    Encodes both semantic and visual knowledge graphs.
    Produces a single fused embedding per class node.

    Outputs:
        sem_emb  : [N, out_dim]  — semantic KG embeddings
        vis_emb  : [N, out_dim]  — visual KG embeddings
        fused    : [N, out_dim]  — combined embedding
    """

    def __init__(
        self,
        sem_in_dim: int,
        vis_in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False,
        gat_heads: int = 4,
        fusion: str = "add",   # add | concat | gate
    ):
        super().__init__()
        self.fusion = fusion

        EncoderCls = GATEncoder if use_gat else GCNEncoder

        if use_gat:
            self.sem_encoder = GATEncoder(
                sem_in_dim, hidden_dim // gat_heads, out_dim,
                heads=gat_heads, dropout=dropout
            )
            self.vis_encoder = GATEncoder(
                vis_in_dim, hidden_dim // gat_heads, out_dim,
                heads=gat_heads, dropout=dropout
            )
        else:
            self.sem_encoder = GCNEncoder(
                sem_in_dim, hidden_dim, out_dim, n_layers, dropout
            )
            self.vis_encoder = GCNEncoder(
                vis_in_dim, hidden_dim, out_dim, n_layers, dropout
            )

        if fusion == "concat":
            self.fusion_proj = nn.Linear(out_dim * 2, out_dim)
        elif fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(out_dim * 2, out_dim)

        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        sem_x: torch.Tensor, sem_edge_index: torch.Tensor,
        vis_x: torch.Tensor, vis_edge_index: torch.Tensor,
    ) -> dict:
        sem_emb = self.sem_encoder(sem_x, sem_edge_index)   # [N, D]
        vis_emb = self.vis_encoder(vis_x, vis_edge_index)   # [N, D]

        if self.fusion == "add":
            fused = sem_emb + vis_emb
        elif self.fusion == "concat":
            fused = self.fusion_proj(torch.cat([sem_emb, vis_emb], dim=-1))
        elif self.fusion == "gate":
            cat = torch.cat([sem_emb, vis_emb], dim=-1)
            gate = self.gate(cat)
            fused = gate * sem_emb + (1 - gate) * vis_emb

        fused = self.out_norm(fused)
        return {
            "sem": sem_emb,
            "vis": vis_emb,
            "fused": fused,
        }


# ── Fallback: no PyG ─────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """
    Fallback encoder when torch-geometric is not installed.
    Applies a 2-layer MLP to node features (ignores graph topology).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        return self.net(x)


def build_kg_encoder(cfg: dict) -> nn.Module:
    """Factory function: build DualKGEncoder or fallback from config dict."""
    if not PYG_OK:
        print("[KGEncoder] torch-geometric not found — using MLP fallback")
        return SimpleMLP(cfg.get("sem_in_dim", 512), cfg["hidden_dim"], cfg["out_dim"])

    return DualKGEncoder(
        sem_in_dim=cfg.get("sem_in_dim", 512),
        vis_in_dim=cfg.get("vis_in_dim", 512),
        hidden_dim=cfg["hidden_dim"],
        out_dim=cfg["out_dim"],
        n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.1),
        use_gat=cfg.get("use_gat", False),
        gat_heads=cfg.get("gat_heads", 4),
        fusion=cfg.get("fusion_method", "add"),
    )


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N, D = 45, 512   # 45 RESISC45 classes, 512-dim features

    # Random graph data
    x_sem = torch.randn(N, D)
    x_vis = torch.randn(N, D)
    # Random edges (sparse)
    edges = torch.randint(0, N, (2, N * 3))

    model = DualKGEncoder(
        sem_in_dim=D, vis_in_dim=D,
        hidden_dim=256, out_dim=512,
        n_layers=2, use_gat=False,
    )
    out = model(x_sem, edges, x_vis, edges)
    print("sem:", out["sem"].shape)   # [45, 512]
    print("vis:", out["vis"].shape)   # [45, 512]
    print("fused:", out["fused"].shape)  # [45, 512]
