"""Small feature-level baselines; neither model updates the visual encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RetrievalModel(nn.Module):
    def __init__(self, video_dim: int, text_dim: int, embed_dim: int,
                 hidden_dim: int, model_name: str) -> None:
        super().__init__()
        self.config = dict(model_name=model_name, video_dim=video_dim,
                           text_dim=text_dim, embed_dim=embed_dim,
                           hidden_dim=hidden_dim)
        self.temporal = (nn.GRU(video_dim, hidden_dim, batch_first=True)
                         if model_name == "gru" else None)
        if model_name == "mlp":
            # Match the GRU video-branch parameter count to within rounding of
            # one hidden unit, while remaining exactly permutation-invariant.
            gru_parameters = 3 * hidden_dim * (video_dim + hidden_dim + 2)
            target = gru_parameters + hidden_dim * embed_dim + embed_dim
            width = max(1, round((target - embed_dim) / (video_dim + embed_dim + 1)))
            self.video_projection = nn.Sequential(nn.Linear(video_dim, width), nn.GELU(),
                                                  nn.Linear(width, embed_dim))
        else:
            self.video_projection = nn.Linear(hidden_dim if self.temporal else video_dim,
                                              embed_dim)
        self.text_projection = nn.Linear(text_dim, embed_dim)

    def encode_video(self, video_features: torch.Tensor) -> torch.Tensor:
        if video_features.ndim != 3:
            raise ValueError("video_features must have shape [batch, frames, dimension]")
        if self.temporal is None:
            pooled = video_features.mean(dim=1)
        else:
            _, hidden = self.temporal(video_features)
            pooled = hidden[-1]
        return F.normalize(self.video_projection(pooled), dim=-1)

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.ndim != 2:
            raise ValueError("text_features must have shape [batch, dimension]")
        return F.normalize(self.text_projection(text_features), dim=-1)

    def forward(self, video_features: torch.Tensor,
                text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_video(video_features), self.encode_text(text_features)


def build_model(model_name: str, video_dim: int, text_dim: int,
                embed_dim: int = 32, hidden_dim: int = 64) -> RetrievalModel:
    if model_name not in {"mean", "mlp", "gru"}:
        raise ValueError("model_name must be 'mean', 'mlp' or 'gru'")
    if min(video_dim, text_dim, embed_dim, hidden_dim) < 1:
        raise ValueError("All model dimensions must be positive")
    return RetrievalModel(video_dim, text_dim, embed_dim, hidden_dim, model_name)
