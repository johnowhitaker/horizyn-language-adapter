"""The whole adapter: optional centering, a linear map (or MLP), normalization."""

import torch
from torch import nn
from torch.nn import functional as F


class Adapter(nn.Module):
    def __init__(self, input_dim, output_dim=512, kind="linear", hidden_dim=1024):
        super().__init__()
        self.config = dict(
            input_dim=input_dim, output_dim=output_dim, kind=kind, hidden_dim=hidden_dim
        )
        self.register_buffer("center", torch.zeros(input_dim))
        self.net = (
            nn.Linear(input_dim, output_dim)
            if kind == "linear"
            else nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
            )
        )

    def forward(self, x):
        x = F.normalize(x - self.center, dim=-1)
        return F.normalize(self.net(x), dim=-1)


def load_adapter(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = Adapter(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint
