import torch
torch.set_default_dtype(torch.float16)
import torch.nn as nn
import torch.nn.functional as F

class TopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_sae)))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_in)))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Initialize decoder weights to have unit norm columns
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1) 
            # Note: usually dimension 1 because shape is (d_sae, d_in)? Or (d_in, d_sae)?
            # If W_dec is (d_sae, d_in), rows are features?
            # Usually x_reconstruct = acts @ W_dec
            # If acts is (batch, d_sae), then W_dec must be (d_sae, d_in).
            # So rows are features. We want features to have unit norm?
            # Or columns if we transpose?
            # Standard: W_dec has shape (d_sae, d_in) if used as acts @ W_dec.
            # Then each row of W_dec is a feature direction in d_in space.
            # So we normalize rows.

            self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1) 

    def encode(self, x):
        # Pre-process: subtract decoder bias (centering)
        x_centered = x - self.b_dec
        
        # Linear
        pre_acts = x_centered @ self.W_enc + self.b_enc
        
        # ReLU
        return F.relu(pre_acts)

    def forward(self, x):
        # Encode
        z = self.encode(x)
        
        # TopK
        # Get top k values and indices
        topk_vals, topk_inds = torch.topk(z, k=self.k, dim=-1)
        
        # Create sparse activation tensor
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_inds, topk_vals)
        
        # Decode
        x_reconstruct = z_sparse @ self.W_dec + self.b_dec
        
        return x_reconstruct, z_sparse
