import torch
import torch.nn as nn


class BottleneckMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1700,
        latent_dim: int = 1024,
        hidden_dim: int = 1024,
        n_layers: int = 6,
    ):
        super().__init__()
        assert n_layers >= 2, "n_layers must be at least 2"

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return self.net(x.reshape(B, -1))
