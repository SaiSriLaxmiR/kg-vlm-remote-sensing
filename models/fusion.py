"""
models/fusion.py
Fusion strategies for injecting knowledge graph embeddings into prompt tokens.
Supports: additive, concatenation, cross-attention gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdditiveFusion(nn.Module):
    """
    Simple element-wise addition of KG embeddings to prompt tokens.
    Requires KG dim == prompt dim (or a projection).
    """

    def __init__(self, kg_dim: int, prompt_dim: int):
        super().__init__()
        self.proj = nn.Linear(kg_dim, prompt_dim, bias=False) if kg_dim != prompt_dim else nn.Identity()
        self.norm = nn.LayerNorm(prompt_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learnable injection weight

    def forward(
        self,
        prompt_tokens: torch.Tensor,   # [N, n_ctx, D]
        kg_emb: torch.Tensor,          # [N, D_kg]
    ) -> torch.Tensor:
        kg = self.proj(kg_emb).unsqueeze(1)   # [N, 1, D]
        out = prompt_tokens + self.alpha * kg
        return self.norm(out)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention: prompt tokens attend to KG node embeddings.
    Allows each context token to selectively pull relevant KG knowledge.

    Query  = prompt context tokens  [N, n_ctx, D]
    Key/Value = KG embeddings       [N, D_kg] → [N, 1, D]
    """

    def __init__(
        self,
        prompt_dim: int,
        kg_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.kg_proj = nn.Linear(kg_dim, prompt_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=prompt_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(prompt_dim)
        self.ff = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim * 4),
            nn.GELU(),
            nn.Linear(prompt_dim * 4, prompt_dim),
        )
        self.ff_norm = nn.LayerNorm(prompt_dim)

    def forward(
        self,
        prompt_tokens: torch.Tensor,  # [N, n_ctx, D]
        kg_emb: torch.Tensor,         # [N, D_kg]
    ) -> torch.Tensor:
        kv = self.kg_proj(kg_emb).unsqueeze(1)  # [N, 1, D]
        attn_out, _ = self.attn(prompt_tokens, kv, kv)
        x = self.norm(prompt_tokens + attn_out)
        x = self.ff_norm(x + self.ff(x))
        return x


class GatedFusion(nn.Module):
    """
    Gated fusion: a sigmoid gate controls how much KG to inject.
    Gate is conditioned on both the prompt token and the KG embedding.
    """

    def __init__(self, prompt_dim: int, kg_dim: int):
        super().__init__()
        self.kg_proj = nn.Linear(kg_dim, prompt_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(prompt_dim * 2, prompt_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(prompt_dim)

    def forward(
        self,
        prompt_tokens: torch.Tensor,  # [N, n_ctx, D]
        kg_emb: torch.Tensor,         # [N, D_kg]
    ) -> torch.Tensor:
        kg = self.kg_proj(kg_emb).unsqueeze(1).expand_as(prompt_tokens)
        gate_input = torch.cat([prompt_tokens, kg], dim=-1)
        gate = self.gate_net(gate_input)
        fused = prompt_tokens + gate * kg
        return self.norm(fused)


class DualFusion(nn.Module):
    """
    Separately fuses semantic and visual KG embeddings into prompt tokens.
    First applies semantic fusion, then visual fusion.
    """

    def __init__(
        self,
        prompt_dim: int,
        kg_dim: int = 512,
        method: str = "add",   # add | cross_attention | gate
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        def make_fusion():
            if method == "cross_attention":
                return CrossAttentionFusion(prompt_dim, kg_dim, n_heads, dropout)
            elif method == "gate":
                return GatedFusion(prompt_dim, kg_dim)
            else:
                return AdditiveFusion(kg_dim, prompt_dim)

        self.sem_fusion = make_fusion()
        self.vis_fusion = make_fusion()
        self.out_norm = nn.LayerNorm(prompt_dim)

    def forward(
        self,
        prompt_tokens: torch.Tensor,        # [N, n_ctx, D]
        sem_kg_emb: Optional[torch.Tensor],  # [N, D]
        vis_kg_emb: Optional[torch.Tensor],  # [N, D]
    ) -> torch.Tensor:
        x = prompt_tokens
        if sem_kg_emb is not None:
            x = self.sem_fusion(x, sem_kg_emb)
        if vis_kg_emb is not None:
            x = self.vis_fusion(x, vis_kg_emb)
        return self.out_norm(x)


def build_fusion(method: str, prompt_dim: int, kg_dim: int, **kwargs) -> nn.Module:
    """Factory for fusion modules."""
    if method == "dual":
        return DualFusion(prompt_dim, kg_dim, **kwargs)
    elif method == "cross_attention":
        return CrossAttentionFusion(prompt_dim, kg_dim, **kwargs)
    elif method == "gate":
        return GatedFusion(prompt_dim, kg_dim)
    else:
        return AdditiveFusion(kg_dim, prompt_dim)


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N, n_ctx, D, kg_D = 45, 16, 512, 512

    prompt = torch.randn(N, n_ctx, D)
    kg_emb = torch.randn(N, kg_D)

    for method in ["add", "cross_attention", "gate"]:
        fusion = DualFusion(D, kg_D, method=method)
        out = fusion(prompt, kg_emb, kg_emb)
        print(f"{method}: {out.shape}")  # [45, 16, 512]
