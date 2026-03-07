import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
import einops

class SAETrainer:
    def __init__(self, sae, model, data_loader, layer, lr=1e-3, 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 aux_loss_weight=1/32, dead_neuron_window=1000):
        self.sae = sae.to(device)
        self.model = model.to(device)
        self.data_loader = data_loader
        self.layer = layer
        self.lr = lr
        self.device = device
        self.aux_loss_weight = aux_loss_weight
        self.dead_neuron_window = dead_neuron_window
        self.optimizer = optim.Adam(self.sae.parameters(), lr=lr)
        
    def _remove_parallel_grad_component(self):
        """
        Remove the component of the decoder gradient that is parallel to the 
        decoder weight directions. This accounts for the interaction between
        unit-norm decoder constraint and Adam optimizer.
        """
        with torch.no_grad():
            W_dec = self.sae.W_dec
            if W_dec.grad is not None:
                # Each row of W_dec is a feature direction
                # Remove the component of the gradient parallel to each row
                # parallel_component = (grad · w_hat) * w_hat
                dec_normed = F.normalize(W_dec.data, p=2, dim=1)
                # Dot product of each grad row with its corresponding weight row
                dot = (W_dec.grad * dec_normed).sum(dim=1, keepdim=True)
                W_dec.grad -= dot * dec_normed

    def _update_dead_neuron_stats(self, z_sparse):
        """Track which neurons are firing."""
        with torch.no_grad():
            # Count how many tokens activated each neuron
            active = (z_sparse > 0).float().sum(dim=0)
            # Increment ticks for all neurons
            self.sae.ticks_since_active += 1
            # Reset ticks for neurons that fired
            self.sae.ticks_since_active[active > 0] = 0
            
            self.sae.total_steps += 1

    def train_step(self, batch_tokens):
        # Unpack batch if it's a list (from TensorDataset)
        if isinstance(batch_tokens, (list, tuple)):
            batch_tokens = batch_tokens[0]
             
        # 1. Get activations from the transformer
        act_name = f"blocks.{self.layer}.hook_resid_post"
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens, 
                stop_at_layer=self.layer + 1,
                names_filter=act_name,
            )
            acts = cache[act_name]  # [batch, seq_len, d_model]
            acts = einops.rearrange(acts, "b s d -> (b s) d")
            # Cast to float32 for SAE training (model may run in float16)
            acts = acts.to(dtype=torch.float32, device=self.device)
            
        # 2. Forward pass through SAE
        self.optimizer.zero_grad()
        recons, z_sparse = self.sae(acts)
        
        # 3. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recons, acts)
        
        # 4. Auxiliary loss for dead neurons
        aux_loss = self.sae.get_auxiliary_loss(acts, z_sparse)
        
        # 5. Total loss
        total_loss = recon_loss + self.aux_loss_weight * aux_loss
        
        total_loss.backward()
        
        # 6. Remove parallel gradient component before optimizer step
        self._remove_parallel_grad_component()
        
        self.optimizer.step()
        
        # 7. Normalize decoder weights to unit norm (rows)
        with torch.no_grad():
            self.sae.W_dec.data = F.normalize(self.sae.W_dec.data, p=2, dim=1)
             
        # 8. Update dead neuron statistics
        self._update_dead_neuron_stats(z_sparse)
        
        return recon_loss.item(), aux_loss.item()

    def train(self, num_epochs=1):
        self.sae.train()
        for epoch in range(num_epochs):
            total_recon = 0
            total_aux = 0
            count = 0
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                recon_loss, aux_loss = self.train_step(batch)
                total_recon += recon_loss
                total_aux += aux_loss
                count += 1
                pbar.set_postfix({
                    "recon": f"{recon_loss:.4f}", 
                    "aux": f"{aux_loss:.4f}",
                    "dead": f"{(self.sae.ticks_since_active > 1000).sum().item()}"
                })
            
            avg_recon = total_recon / count if count > 0 else 0
            avg_aux = total_aux / count if count > 0 else 0
            print(f"Epoch {epoch+1} — Avg Recon Loss: {avg_recon:.4f}, Avg Aux Loss: {avg_aux:.4f}")
        
        return self.sae
