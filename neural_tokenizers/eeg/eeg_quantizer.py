import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE: the neural_tokenizers/CLAUDE.md §4 Stage-1 spec called for `memcodes`
# from fourm.vq.quantizers.  This is a straight-through product-VQ instead.
# The two behave similarly (nearest-neighbour lookup + commitment loss) but
# memcodes uses EMA codebook updates while this uses gradient descent on the
# embedding table.  The Stage-1 checkpoint was trained with this class, so
# swapping quantizers would require a re-train.  Since Stage 1 is the expected-
# to-fail baseline and LaBraM (Stage 2c) is the production tokenizer, leaving
# this as-is is acceptable; flag as tech-debt if a Stage-1 retrain is ever done.
class ProductVQQuantizer(nn.Module):
    def __init__(
        self,
        latent_dim: int = 1024,
        codebook_size: int = 1024,
        num_codebooks: int = 8,
    ):
        super().__init__()
        assert latent_dim % num_codebooks == 0, "latent_dim must be divisible by num_codebooks"
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.sub_dim = latent_dim // num_codebooks

        self.codebooks = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, self.sub_dim) * 0.01
        )

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        # (B, num_codebooks, sub_dim)
        chunks = z.reshape(B, self.num_codebooks, self.sub_dim)

        # distances: (B, num_codebooks, codebook_size)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z_sq = (chunks ** 2).sum(-1, keepdim=True)           # (B, K, 1)
        e_sq = (self.codebooks ** 2).sum(-1).unsqueeze(0)    # (1, K, V)
        dot = torch.einsum("bkd,kvd->bkv", chunks, self.codebooks)  # (B, K, V)
        dist = z_sq + e_sq - 2 * dot                         # (B, K, V)

        codes = dist.argmin(dim=-1)  # (B, num_codebooks)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        B = codes.shape[0]
        # codes: (B, num_codebooks)
        # codebooks: (num_codebooks, codebook_size, sub_dim)
        # For each codebook k, gather the entry at codes[:, k].
        k_idx = torch.arange(self.num_codebooks, device=codes.device)  # (K,)
        # embeddings: (B, K, sub_dim)
        embeddings = self.codebooks[k_idx, codes]  # advanced indexing: codes[b,k] selects from codebooks[k]
        return embeddings.reshape(B, self.latent_dim)

    def forward(self, z: torch.Tensor):
        codes = self.encode(z)
        z_q = self.decode(codes)
        # Straight-through: encoder gradients flow, decoder gets z_q_st.
        z_q_st = z + (z_q - z).detach()
        # Both terms: codebook moves toward encoder outputs, encoder moves toward codebook.
        vq_loss = F.mse_loss(z.detach(), z_q) + 0.25 * F.mse_loss(z, z_q.detach())
        return z_q_st, vq_loss, codes

    def tokens_to_embedding(self, codes: torch.Tensor) -> torch.Tensor:
        B = codes.shape[0]
        k_idx = torch.arange(self.num_codebooks, device=codes.device)
        return self.codebooks[k_idx, codes]  # (B, num_codebooks, sub_dim)
