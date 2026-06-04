import torch
import torch.nn as nn


class BottleneckMLPDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        output_dim: int = 1700,
        hidden_dim: int = 1024,
        n_layers: int = 6,
        signal_shape: tuple[int, int] = (17, 100),
    ):
        super().__init__()
        assert n_layers >= 2, "n_layers must be at least 2"
        self.signal_shape = signal_shape

        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return self.net(z).reshape(B, *self.signal_shape)
