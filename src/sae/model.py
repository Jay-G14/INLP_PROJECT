import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        # Decoder weights: shape (d_sae, d_in)
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_in))
        )
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Normalize decoder rows to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

        # Encoder weights: tied initialization (transpose of decoder)
        # This helps prevent dead latents and improves MSE
        self.W_enc = nn.Parameter(self.W_dec.data.T.clone())  # shape (d_in, d_sae)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Track dead neurons
        self.register_buffer("ticks_since_active", torch.zeros(d_sae))
        self.register_buffer("total_steps", torch.tensor(0))

    def encode(self, x):
        """Compute pre-activations (before TopK). No ReLU — TopK IS the activation."""
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        # NO ReLU here. TopK is applied in forward() and serves as the activation function.
        return pre_acts

    def forward(self, x):
        pre_acts = self.encode(x)
        topk_vals, topk_inds = torch.topk(pre_acts, k=self.k, dim=-1)
        topk_vals = F.relu(topk_vals)

        z_sparse = torch.zeros_like(pre_acts)
        z_sparse.scatter_(-1, topk_inds, topk_vals)
        x_reconstruct = z_sparse @ self.W_dec + self.b_dec

        return x_reconstruct, z_sparse

    def get_auxiliary_loss(self, x, z_sparse, num_dead_sample=32):
        """
        Auxiliary loss to revive dead neurons.
        Uses the reconstruction error to train dead neurons.
        """
        # Identify dead neurons (not activated in recent steps)
        if self.total_steps < 100:
            return torch.tensor(0.0, device=x.device)

        # A neuron is dead if it hasn't fired in the last 1000 steps
        dead_mask = self.ticks_since_active > 1000
        num_dead = dead_mask.sum().item()

        if num_dead == 0:
            return torch.tensor(0.0, device=x.device)

        x_reconstruct = z_sparse @ self.W_dec + self.b_dec
        error = x - x_reconstruct
        dead_pre_acts = (error - self.b_dec) @ self.W_enc + self.b_enc
        dead_pre_acts_masked = dead_pre_acts.clone()
        dead_pre_acts_masked[:, ~dead_mask] = float("-inf")

        # TopK among dead neurons only
        k_dead = min(num_dead_sample, num_dead)
        topk_vals, topk_inds = torch.topk(dead_pre_acts_masked, k=k_dead, dim=-1)
        topk_vals = F.relu(topk_vals)

        z_dead = torch.zeros_like(dead_pre_acts)
        z_dead.scatter_(-1, topk_inds, topk_vals)
        error_recon = z_dead @ self.W_dec
        aux_loss = F.mse_loss(error_recon, error.detach())

        return aux_loss
