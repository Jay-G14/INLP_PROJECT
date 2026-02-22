import torch
torch.set_default_dtype(torch.float16)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
import einops

class SAETrainer:
    def __init__(self, sae, model, data_loader, layer, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.sae = sae.to(device)
        self.model = model.to(device)
        self.data_loader = data_loader
        self.layer = layer
        self.lr = lr
        self.device = device
        self.optimizer = optim.Adam(self.sae.parameters(), lr=lr)
        
    def train_step(self, batch_tokens):
        # Unpack batch if it's a list (from TensorDataset)
        if isinstance(batch_tokens, list) or isinstance(batch_tokens, tuple):
             batch_tokens = batch_tokens[0]
             
        # 1. Get activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(batch_tokens, stop_at_layer=self.layer + 1)
            # Assuming we want the residual stream or MLP output
            # Layer 6, 8, 10
            # Target identifying features in Residual Stream?
            # Outline: "Residual Stream and MLP Outputs at Layers 6, 8, and 10"
            # Let's target the residual stream after the MLP: blocks.{layer}.hook_resid_post
            # Or MLP output: blocks.{layer}.mlp.hook_post
            
            act_name = f"blocks.{self.layer}.hook_resid_post" 
            # or blocks.{layer}.mlp.hook_post
            
            # Let's use resid_post for now as it captures everything up to that point.
            acts = cache[act_name] 
            # shape: [batch, seq_len, d_model]
            
            # Flatten batch and seq_len
            acts = einops.rearrange(acts, "b s d -> (b s) d")
            
        # 2. Train SAE
        self.optimizer.zero_grad()
        
        # Forward pass
        recons, sparse_acts = self.sae(acts)
        
        # Loss: reconstruction MSE
        # TopK SAE usually just minimizes MSE, as sparsity is enforced by TopK.
        loss = F.mse_loss(recons, acts)
        
        loss.backward()
        self.optimizer.step()
        
        # 3. Normalize decoder weights (optional but recommended for SAEs)
        # Some implementations do this every step
        with torch.no_grad():
             self.sae.W_dec.data = F.normalize(self.sae.W_dec.data, p=2, dim=1)
             
        return loss.item()

    def train(self, num_epochs=1):
        self.sae.train()
        for epoch in range(num_epochs):
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                loss = self.train_step(batch)
                pbar.set_postfix({"loss": loss})
        
        return self.sae
