import torch
torch.set_default_dtype(torch.float16)
from jaxtyping import Float

def get_ablation_hook(sae, feature_indices_to_ablate, clamp_value=-20.0):
    """
    Returns a hook function that intervenes on specific SAE features.
    
    Uses negative clamping (from inspiration paper) instead of zeroing:
    - When a target feature activates (> 0), clamp it to a negative value
    - This preserves model perplexity better than zeroing
    
    Args:
        sae: The trained SAE model
        feature_indices_to_ablate: List/tensor of feature indices to intervene on
        clamp_value: Negative value to clamp activating features to (default: -20)
    """
    def hook(activation: Float[torch.Tensor, "batch seq d_model"], hook):
        # Activation shape: [batch, seq, d_model]
        
        # Clone to avoid in-place errors
        original_act = activation.clone()
        
        # Get SAE reconstruction and sparse activations
        x_reconstruct, z_sparse = sae(activation)
        
        # Clone for modification
        z_modified = z_sparse.clone()
        
        # NEGATIVE CLAMPING (inspiration paper approach):
        # Only clamp features that are activating (> 0) to negative value
        # This preserves more model capability than zeroing
        target_features = z_sparse[:, :, feature_indices_to_ablate]
        mask = target_features > 0
        
        # Apply clamping: set to clamp_value where feature activates, else keep at 0
        z_modified[:, :, feature_indices_to_ablate] = torch.where(
            mask,
            torch.full_like(target_features, clamp_value),
            target_features  # Keep original (usually 0 from TopK sparsity)
        )
        
        # Decode with modified features
        x_modified_recon = z_modified @ sae.W_dec + sae.b_dec
        
        # Preserve the reconstruction error (what SAE didn't capture)
        error = original_act - x_reconstruct
        modified_act = x_modified_recon + error
        
        return modified_act
        
    return hook
