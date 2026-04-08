import math
from typing import Any, Dict, Optional

import einops
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.sae.checkpoints import save_training_checkpoint


class SAETrainer:
    def __init__(
        self,
        sae,
        model,
        data_loader,
        layer,
        lr=1e-4,
        model_device="cuda",
        sae_device="cpu",
        aux_loss_weight=1 / 32,
        dead_neuron_window=1000,
        model_family="gpt2",
        max_grad_norm=1.0,
        checkpoint_path: Optional[str] = None,
        checkpoint_args: Optional[Dict[str, Any]] = None,
    ):
        self.sae = sae.to(sae_device)
        self.model = model
        self.data_loader = data_loader
        self.layer = layer
        self.lr = lr
        self.model_device = model_device
        self.sae_device = sae_device
        self.aux_loss_weight = aux_loss_weight
        self.dead_neuron_window = dead_neuron_window
        self.optimizer = optim.Adam(self.sae.parameters(), lr=lr)
        self.model_family = model_family
        self.max_grad_norm = max_grad_norm
        self.checkpoint_path = checkpoint_path
        self.checkpoint_args = checkpoint_args or {}

    def _remove_parallel_grad_component(self):
        with torch.no_grad():
            w_dec = self.sae.W_dec
            if w_dec.grad is not None:
                dec_normed = F.normalize(w_dec.data, p=2, dim=1)
                dot = (w_dec.grad * dec_normed).sum(dim=1, keepdim=True)
                w_dec.grad -= dot * dec_normed

    def _update_dead_neuron_stats(self, z_sparse):
        with torch.no_grad():
            active = (z_sparse > 0).float().sum(dim=0)
            self.sae.ticks_since_active += 1
            self.sae.ticks_since_active[active > 0] = 0
            self.sae.total_steps += 1

    def _get_acts(self, batch_tokens):
        with torch.no_grad():
            if self.model_family == "gpt2":
                _, cache = self.model.run_with_cache(
                    batch_tokens.to(self.model_device),
                    stop_at_layer=self.layer + 1,
                )
                acts = cache[f"blocks.{self.layer}.hook_resid_post"]
            else:
                outputs = self.model(
                    batch_tokens.to(self.model_device),
                    output_hidden_states=True,
                    use_cache=False,
                )
                acts = outputs.hidden_states[self.layer + 1]

        acts = acts.float().to(self.sae_device)
        return einops.rearrange(acts, "b s d -> (b s) d")

    def train_step(self, batch_tokens):
        if isinstance(batch_tokens, (list, tuple)):
            batch_tokens = batch_tokens[0]

        acts = self._get_acts(batch_tokens)
        self.optimizer.zero_grad(set_to_none=True)

        recons, z_sparse = self.sae(acts)
        recon_loss = F.mse_loss(recons, acts)
        aux_loss = self.sae.get_auxiliary_loss(acts, z_sparse)
        total_loss = recon_loss + self.aux_loss_weight * aux_loss

        if not torch.isfinite(total_loss):
            del acts, recons, z_sparse, recon_loss, aux_loss, total_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"skipped": True, "recon": float("nan"), "aux": float("nan")}

        total_loss.backward()

        grad_is_finite = True
        for param in self.sae.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                grad_is_finite = False
                break

        if not grad_is_finite:
            self.optimizer.zero_grad(set_to_none=True)
            del acts, recons, z_sparse, recon_loss, aux_loss, total_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"skipped": True, "recon": float("nan"), "aux": float("nan")}

        self._remove_parallel_grad_component()
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            self.sae.W_dec.data = F.normalize(self.sae.W_dec.data, p=2, dim=1)

        self._update_dead_neuron_stats(z_sparse)

        recon_val = recon_loss.item()
        aux_val = aux_loss.item()

        del acts, recons, z_sparse, recon_loss, aux_loss, total_loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"skipped": False, "recon": recon_val, "aux": aux_val}

    def train(
        self,
        num_epochs=5,
        max_steps: Optional[int] = None,
        save_every_steps: Optional[int] = None,
        start_epoch: int = 0,
        start_step_in_epoch: int = 0,
        global_step: int = 0,
    ):
        self.sae.train()
        for epoch in range(start_epoch, num_epochs):
            total_recon = 0.0
            total_aux = 0.0
            count = 0
            skipped_steps = 0
            pbar = tqdm(self.data_loader, desc=f"Epoch {epoch + 1}")

            for step_in_epoch, batch in enumerate(pbar):
                if epoch == start_epoch and step_in_epoch < start_step_in_epoch:
                    continue

                stats = self.train_step(batch)
                if stats["skipped"]:
                    skipped_steps += 1
                else:
                    total_recon += stats["recon"]
                    total_aux += stats["aux"]
                    count += 1

                global_step += 1

                recon_str = "nan" if math.isnan(stats["recon"]) else f"{stats['recon']:.4f}"
                aux_str = "nan" if math.isnan(stats["aux"]) else f"{stats['aux']:.4f}"
                pbar.set_postfix(
                    {
                        "step": global_step,
                        "recon": recon_str,
                        "aux": aux_str,
                        "skipped": skipped_steps,
                    }
                )

                if self.checkpoint_path and save_every_steps and global_step % save_every_steps == 0:
                    save_training_checkpoint(
                        self.checkpoint_path,
                        self.sae,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        step_in_epoch=step_in_epoch + 1,
                        global_step=global_step,
                        args=self.checkpoint_args,
                    )

                if max_steps is not None and global_step >= max_steps:
                    avg_recon = total_recon / count if count > 0 else float("nan")
                    avg_aux = total_aux / count if count > 0 else float("nan")
                    print(
                        f"Epoch {epoch + 1} — Avg Recon Loss: {avg_recon:.4f}, Avg Aux Loss: {avg_aux:.4f}, Skipped: {skipped_steps}"
                    )
                    return {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch + 1,
                        "global_step": global_step,
                    }

            avg_recon = total_recon / count if count > 0 else float("nan")
            avg_aux = total_aux / count if count > 0 else float("nan")
            print(
                f"Epoch {epoch + 1} — Avg Recon Loss: {avg_recon:.4f}, Avg Aux Loss: {avg_aux:.4f}, Skipped: {skipped_steps}"
            )
            start_step_in_epoch = 0

        return {
            "epoch": max(num_epochs - 1, 0),
            "step_in_epoch": 0,
            "global_step": global_step,
        }

