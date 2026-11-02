"""Gated-attention multiple-instance learning (Ilse et al., ICML 2018) over DinoBloom cell features.

One bag = all white cells of one patient sample (or of one field, for field-level predictions).
The attention weights say how much each cell contributed to the bag's diagnosis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CLASSES = ["ALL", "AML", "CLL", "CML", "NO_LEUKEMIA"]


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, att: int = 128, n_classes: int = 5, dropout: float = 0.25):
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.att_v = nn.Sequential(nn.Linear(hidden, att), nn.Tanh())
        self.att_u = nn.Sequential(nn.Linear(hidden, att), nn.Sigmoid())
        self.att_w = nn.Linear(att, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor):
        """x: (n_cells, in_dim) -> logits (n_classes,), attention (n_cells,)"""
        h = self.embed(x)
        a = self.att_w(self.att_v(h) * self.att_u(h)).squeeze(-1)
        a = torch.softmax(a, dim=0)
        z = (a.unsqueeze(-1) * h).sum(0)
        return self.head(z), a
