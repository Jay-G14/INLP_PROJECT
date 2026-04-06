import torch
from jaxtyping import Float

def get_ablation_hook(sae, feature_indices_to_ablate, mean_activations=None, scale=-5.0):
    """
    Returns a hook function that ablates specific features using the SAE.
    
    This uses CONDITIONAL NEGATIVE SCALING: if a targeted feature fires,
    we explicitly invert and amplify its activation (e.g., scale * z_sparse).
    This actively suppresses the feature's contribution ONLY on tokens 
    where it tries to activate, preserving general capabilities (perplexity)
    unlike unconditional steering.
    
    Args:
        sae: The trained Sparse Autoencoder
        feature_indices_to_ablate: Tensor of feature indices to ablate
        mean_activations: (Unused in this updated method, kept for signature compatibility)
        scale: Scaling factor. 0.0 means standard zero-ablation. 
               -5.0 means conditional negative steering.
    """
    def hook(activation: Float[torch.Tensor, "batch seq d_model"], hook):
        # Activation shape: [batch, seq, d_model]
        original_act = activation.clone().to(next(sae.parameters()).device)
        
        # 1. Full forward pass through SAE
        x_reconstruct, z_sparse = sae(activation)
        
        # 2. Ablate
        z_ablated = z_sparse.clone()
        
        if scale == 0.0:
            # Standard zero ablation
            z_ablated[..., feature_indices_to_ablate] = 0.0
        else:
            # Conditional negative scaling
            # Only modify tokens where the feature actually fired
            target_acts = z_sparse[..., feature_indices_to_ablate]
            fired_mask = target_acts > 0
            
            z_ablated[..., feature_indices_to_ablate] = torch.where(
                fired_mask,
                target_acts * scale,
                0.0
            )
        
        # 3. Decode the ablated activations
        x_ablated_recon = z_ablated @ sae.W_dec + sae.b_dec
        
        # 4. Preserve the SAE reconstruction error
        error = original_act - x_reconstruct
        modified_act = x_ablated_recon + error
        
        return modified_act
        
    return hook
