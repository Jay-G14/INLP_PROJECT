import torch
from jaxtyping import Float

def get_ablation_hook(sae, feature_indices_to_ablate):
    """
    Returns a hook function that ablates specific features using the SAE.
    """
    def hook(activation: Float[torch.Tensor, "batch seq d_model"], hook):
        # Activation shape: [batch, seq, d_model]
        
        # 1. Encode
        # SAE expects [batch, d_model] usually, but handled batch/seq flattening in training?
        # My TopKSAE implementation takes [..., d_in] and returns [..., d_out].
        # It handles arbitrary leading dimensions if implemented with standard linear layers.
        # Let's check model.py:
        # encode(x): (x - b_dec) @ W_enc + b_enc. Yes, handles B, S.
        
        # We need to clone activation to avoid in-place errors if needed, but usually fine.
        original_act = activation.clone()
        
        x_reconstruct, z_sparse = sae(activation)
        
        # 2. Ablate
        # z_sparse: [batch, seq, d_sae] (after TopK and scatter)
        # We want to ZERO out the features in z_sparse that correspond to Harry Potter.
        
        # Create a mask or just zero them out
        # feature_indices_to_ablate is a list or tensor of indices.
        
        z_ablated = z_sparse.clone()
        z_ablated[:, :, feature_indices_to_ablate] = 0.0
        
        # 3. Decode
        # We need to use the DECODER of the SAE to get the ablated reconstruction.
        # x_ablated_recon = z_ablated @ W_dec + b_dec
        x_ablated_recon = z_ablated @ sae.W_dec + sae.b_dec
        
        # 4. Reconstruction Error
        # We want to preserve the information that the SAE *didn't* capture.
        # error = original_act - x_reconstruct
        # new_act = x_ablated_recon + error
        # This is equivalent to: original_act - (x_reconstruct - x_ablated_recon)
        # i.e. removing the contribution of the specific features.
        
        error = original_act - x_reconstruct
        modified_act = x_ablated_recon + error
        
        return modified_act
        
    return hook
