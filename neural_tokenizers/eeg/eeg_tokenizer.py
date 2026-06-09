import torch
import torch.nn as nn
import torch.nn.functional as F

from .eeg_encoder import BottleneckMLPEncoder
from .eeg_decoder import BottleneckMLPDecoder
from .eeg_quantizer import ProductVQQuantizer


class Stage1EEGTokenizer(nn.Module):
    codebook_size: int = 1024

    def __init__(
        self,
        channels: int = 17,
        time: int = 100,
        latent_dim: int = 1024,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        hidden_dim: int = 1024,
        n_layers: int = 6,
    ):
        super().__init__()
        self.channels = channels
        self.time = time
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        input_dim = channels * time
        self.encoder = BottleneckMLPEncoder(input_dim, latent_dim, hidden_dim, n_layers)
        self.quantizer = ProductVQQuantizer(latent_dim, codebook_size, num_codebooks)
        self.decoder = BottleneckMLPDecoder(latent_dim, input_dim, hidden_dim, n_layers, (channels, time))

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        codes = self.quantizer.encode(z)
        return codes

    def decode_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        z_q = self.quantizer.decode(codes)
        return self.decoder(z_q)

    def tokens_to_embedding(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.tokens_to_embedding(codes)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q_st, commit_loss, codes = self.quantizer(z)
        x_hat = self.decoder(z_q_st)
        recon_loss = F.smooth_l1_loss(x_hat, x)
        total_loss = recon_loss + commit_loss
        return x_hat, total_loss, codes

    def save(self, path: str) -> None:
        linears = [m for m in self.encoder.net if isinstance(m, nn.Linear)]
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "channels": self.channels,
                    "time": self.time,
                    "latent_dim": linears[-1].out_features,
                    "num_codebooks": self.num_codebooks,
                    "codebook_size": self.codebook_size,
                    "hidden_dim": linears[0].out_features,
                    "n_layers": len(linears),
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "Stage1EEGTokenizer":
        obj = torch.load(path, map_location="cpu")
        model = cls(**obj["config"])
        model.load_state_dict(obj["state_dict"])
        return model
