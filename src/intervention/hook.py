import torch


def get_ablation_hook(
    sae,
    feature_indices_to_ablate,
    mean_activations=None,
    scale=-5.0,
    hook_stats=None,
):
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

    feature_indices = torch.as_tensor(
        feature_indices_to_ablate, dtype=torch.long, device=sae.W_dec.device
    )

    def hook(activation: torch.Tensor, *args, **kwargs):
        # Activation shape: [batch, seq, d_model]
        original_device = activation.device
        original_dtype = activation.dtype
        sae_device = sae.W_dec.device
        activation = activation.to(device=sae_device, dtype=torch.float32)

        # forward pass through SAE
        x_reconstruct, z_sparse = sae(activation)

        z_ablated = z_sparse.clone()

        if scale == 0.0:
            z_ablated[..., feature_indices] = 0.0
        else:
            # Only modify tokens where the feature actually fired
            target_acts = z_sparse[..., feature_indices]
            fired_mask = target_acts > 0

            if hook_stats is not None:
                hook_stats["total_target_positions"] += int(target_acts.numel())
                hook_stats["fired_target_positions"] += int(fired_mask.sum().item())
                hook_stats["target_activation_sum"] += float(
                    target_acts[fired_mask].sum().item() if fired_mask.any() else 0.0
                )

            z_ablated[..., feature_indices] = torch.where(
                fired_mask, target_acts * scale, 0.0
            )

        # decode the ablated activations
        x_ablated_recon = z_ablated @ sae.W_dec + sae.b_dec

        # preserve the SAE reconstruction error
        error = activation - x_reconstruct
        modified_act = x_ablated_recon + error

        return modified_act.to(device=original_device, dtype=original_dtype)

    return hook
