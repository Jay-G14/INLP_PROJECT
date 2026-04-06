import torch
from jaxtyping import Float

def get_ablation_hook(sae, feature_indices_to_ablate, mean_activations=None, scale=0.0):
    """
    Returns a hook function that ablates specific features using the SAE.
    
    Zero-ablation: targeted features are set to 0, cleanly removing their 
    contribution from the residual stream while preserving all other features
    and the SAE reconstruction error.
    
    Args:
        sae: The trained Sparse Autoencoder
        feature_indices_to_ablate: Tensor of feature indices to ablate
        mean_activations: (Unused, kept for signature compatibility)
        scale: (Unused, kept for signature compatibility. Always zero-ablates.)
    """
    def hook(activation: Float[torch.Tensor, "batch seq d_model"], hook):
        # Activation shape: [batch, seq, d_model]
        original_act = activation.clone().to(next(sae.parameters()).device)
        
        # 1. Full forward pass through SAE
        x_reconstruct, z_sparse = sae(activation)
        
        # 2. Zero-ablation: cleanly remove targeted features
        z_ablated = z_sparse.clone()
        z_ablated[..., feature_indices_to_ablate] = 0.0
        
        # 3. Decode the ablated activations
        x_ablated_recon = z_ablated @ sae.W_dec + sae.b_dec
        
        # 4. Preserve the SAE reconstruction error
        error = original_act - x_reconstruct
        modified_act = x_ablated_recon + error
        
        return modified_act
        
    return hook
