import sys
import os
from dotenv import load_dotenv
from huggingface_hub import login

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from src.sae.model import TopKSAE
import einops
import argparse

# ── helpers ────────────────────────────────────────────────────────────────

def load_model(model_name, device):
    if model_name.startswith("meta-llama"):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = HookedTransformer.from_pretrained(
            model_name, hf_model=hf_model, device="cpu",
            fold_ln=False, center_writing_weights=False, center_unembed=False
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name, device=device,
            fold_ln=False, center_writing_weights=False, center_unembed=False
        )
    return model


def get_activations(model, sae, text, layer, device):
    """Return (acts, recons, z_sparse) for a piece of text."""
    model_device = next(model.parameters()).device
    tokens = model.to_tokens(text).to(model_device)
    act_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda n: n == act_name,
            stop_at_layer=layer + 1
        )
    acts = cache[act_name].reshape(-1, sae.d_in).to(device)
    # Cast acts to the correct dtype for SAE
    acts = acts.to(sae.W_dec.dtype)
    with torch.no_grad():
        recons, z_sparse = sae(acts)
    return acts, recons, z_sparse


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── Test 1: Reconstruction Quality ─────────────────────────────────────────

def test_reconstruction(model, sae, layer, device):
    section("TEST 1: Reconstruction Quality")

    test_texts = [
        "The cat sat on the mat and looked out the window.",
        "Harry picked up his wand and whispered the incantation.",
        "The stock market fell sharply today amid rising inflation fears."
    ]

    all_fve = []
    for text in test_texts:
        acts, recons, z_sparse = get_activations(model, sae, text, layer, device)

        # Handle datatypes correctly for loss
        recons = recons.to(torch.float32)
        acts = acts.to(torch.float32)

        mse      = F.mse_loss(recons, acts).item()
        variance = acts.var().item()
        fve      = 1.0 - (mse / variance)          # fraction of variance explained
        avg_act  = (z_sparse > 0).float().sum(dim=-1).mean().item()
        all_fve.append(fve)

        status = "✅" if fve > 0.80 else ("⚠️ " if fve > 0.60 else "❌")
        print(f"\nText : '{text[:55]}...'")
        print(f"  MSE                       : {mse:.4f}")
        print(f"  Activation variance       : {variance:.4f}")
        print(f"  Fraction variance explained: {fve:.3f}  {status}")
        print(f"  Avg features active/token : {avg_act:.1f}  "
              f"{'✅' if abs(avg_act - sae.k) < 2 else '❌'} (expected ~{sae.k})")

    mean_fve = sum(all_fve) / len(all_fve)
    print(f"\nMean FVE across all texts: {mean_fve:.3f}")
    if mean_fve > 0.90:
        print("→ RESULT: ✅ Excellent reconstruction")
    elif mean_fve > 0.80:
        print("→ RESULT: ✅ Good reconstruction")
    elif mean_fve > 0.60:
        print("→ RESULT: ⚠️  Borderline — SAE may be underfit")
    else:
        print("→ RESULT: ❌ Poor reconstruction — SAE likely broken")
    return mean_fve

# ── Test 2: Sparsity Distribution ──────────────────────────────────────────

def test_sparsity(model, sae, layer, device):
    section("TEST 2: Feature Sparsity Distribution")

    # Use a generic Wikipedia-style paragraph
    text = """
    The history of mathematics covers the development of mathematical ideas
    throughout human history. Early mathematics developed for practical purposes
    such as counting, measurement, and trade. Ancient civilizations including
    the Egyptians, Babylonians, Greeks, and Chinese all contributed to the
    development of mathematics. The Greek mathematicians introduced formal
    proofs and abstract reasoning. During the Islamic Golden Age scholars
    preserved and expanded upon ancient knowledge. The Renaissance saw renewed
    interest in classical learning, and the Scientific Revolution brought new
    mathematical tools including calculus invented independently by Newton and
    Leibniz. Modern mathematics is highly abstract and specialised.
    """

    acts, _, z_sparse = get_activations(model, sae, text, layer, device)
    freq = (z_sparse > 0).float().mean(dim=0)   # [d_sae]

    never     = (freq == 0).sum().item()
    rare      = ((freq > 0)    & (freq < 0.01)).sum().item()
    common    = ((freq >= 0.01) & (freq < 0.05)).sum().item()
    frequent  = ((freq >= 0.05) & (freq < 0.20)).sum().item()
    very_freq = (freq >= 0.20).sum().item()

    print(f"\nFeature firing distribution (d_sae={sae.d_sae}):")
    print(f"  Never fire (dead)      : {never:6,}  ({never/sae.d_sae*100:.1f}%)")
    print(f"  Fire  < 1%  (rare)     : {rare:6,}  ({rare/sae.d_sae*100:.1f}%)")
    print(f"  Fire 1–5%   (moderate) : {common:6,}  ({common/sae.d_sae*100:.1f}%)")
    print(f"  Fire 5–20%  (frequent) : {frequent:6,}  ({frequent/sae.d_sae*100:.1f}%)")
    print(f"  Fire >20%   (generic)  : {very_freq:6,}  ({very_freq/sae.d_sae*100:.1f}%)")

    print(f"\nTop 10 most active features:")
    top_vals, top_idx = torch.topk(freq, 10)
    for i in range(10):
        print(f"  Feature {top_idx[i].item():6d}:  fires {top_vals[i].item()*100:.2f}% of tokens")

    # Verdict
    if never < sae.d_sae * 0.60 and very_freq < 200:
        print("\n→ RESULT: ✅ Healthy sparsity distribution")
    elif never > sae.d_sae * 0.90:
        print("\n→ RESULT: ❌ Too many dead features — SAE likely overfit")
    elif very_freq > 500:
        print("\n→ RESULT: ❌ Too many generic features — features may have collapsed")
    else:
        print("\n→ RESULT: ⚠️  Borderline distribution — check individual features")

    return freq

# ── Test 3: Semantic Feature Overlap ───────────────────────────────────────

def test_semantic_overlap(model, sae, layer, device):
    section("TEST 3: Semantic Feature Overlap")

    def top_features(text, topn=50):
        _, _, z = get_activations(model, sae, text, layer, device)
        scores = z.sum(dim=0)
        _, idx = torch.topk(scores, topn)
        return set(idx.cpu().tolist())

    pairs = [
        (
            "Harry cast a spell with his wand at Hogwarts",
            "The chef carefully prepared the evening meal",
            "HP vs cooking (should differ)"
        ),
        (
            "Hermione read her potions textbook carefully",
            "Ron laughed loudly at the quidditch joke",
            "Two HP sentences (moderate overlap expected)"
        ),
        (
            "The cat sat on the mat by the door",
            "The dog ran quickly across the field",
            "Two generic sentences (should overlap most)"
        ),
    ]

    results = []
    for t1, t2, label in pairs:
        f1, f2 = top_features(t1), top_features(t2)
        overlap = len(f1 & f2) / len(f1 | f2) if len(f1 | f2) > 0 else 0
        results.append((label, overlap))
        print(f"\n{label}")
        print(f"  '{t1[:45]}'")
        print(f"  '{t2[:45]}'")
        print(f"  Overlap: {overlap*100:.1f}%")

    diff_domain = results[0][1]
    same_domain = results[2][1]

    print(f"\nDiff-domain overlap : {diff_domain*100:.1f}%  (want < 30%)")
    print(f"Same-domain overlap : {same_domain*100:.1f}%  (want > 30%)")

    if diff_domain < 0.30 and same_domain > 0.30:
        print("→ RESULT: ✅ Features encode semantic meaning")
    elif diff_domain > 0.50:
        print("→ RESULT: ❌ Features too generic — not domain-sensitive")
    else:
        print("→ RESULT: ⚠️  Unclear — check with more examples")

# ── Test 4: Ablation KL Divergence ─────────────────────────────────────────

def test_ablation(model, sae, layer, device, top_feature_indices):
    section("TEST 4: Ablation KL Divergence")

    def ablation_kl(text, feature_indices):
        model_device = next(model.parameters()).device
        tokens = model.to_tokens(text).to(model_device)

        with torch.no_grad():
            normal_logits = model(tokens)

        def hook_fn(acts, hook):
            flat = acts.reshape(-1, acts.shape[-1]).to(device)
            # Ensure dtype matching
            flat = flat.to(sae.W_dec.dtype)
            with torch.no_grad():
                _, z = sae(flat)
            z_abl = z.clone()
            z_abl[:, feature_indices] = 0.0
            recons_abl = z_abl @ sae.W_dec + sae.b_dec
            return recons_abl.reshape(acts.shape).to(acts.dtype).to(acts.device)

        with torch.no_grad():
            abl_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]
            )

        kl = F.kl_div(
            abl_logits[:, -1].log_softmax(-1),
            normal_logits[:, -1].softmax(-1),
            reduction='sum'
        ).item()
        return kl

    hp_texts = [
        "Harry raised his wand and whispered the spell",
        "Hermione opened her copy of Hogwarts: A History",
    ]
    neutral_texts = [
        "The scientist published the results in the journal",
        "She walked to the shop and bought some bread",
    ]

    feat_tensor = torch.tensor(top_feature_indices, device=device)

    print(f"\nAblating top {len(top_feature_indices)} HP features...\n")
    hp_kls, neu_kls = [], []

    for t in hp_texts:
        kl = ablation_kl(t, feat_tensor)
        hp_kls.append(kl)
        print(f"  HP text KL      : {kl:.4f}  '{t[:45]}'")

    for t in neutral_texts:
        kl = ablation_kl(t, feat_tensor)
        neu_kls.append(kl)
        print(f"  Neutral text KL : {kl:.4f}  '{t[:45]}'")

    avg_hp  = sum(hp_kls)  / len(hp_kls)
    avg_neu = sum(neu_kls) / len(neu_kls)
    ratio   = avg_hp / (avg_neu + 1e-6)

    print(f"\nAvg HP KL      : {avg_hp:.4f}  (want > 1.0)")
    print(f"Avg Neutral KL : {avg_neu:.4f}  (want < 0.1)")
    print(f"Ratio HP/Neutral: {ratio:.1f}x  (want > 5x)")

    if avg_hp > 1.0 and ratio > 5:
        print("→ RESULT: ✅ Features are genuinely HP-specific")
    elif avg_hp < 0.1:
        print("→ RESULT: ❌ Ablation has no effect — features are noise")
    else:
        print("→ RESULT: ⚠️  Weak signal — features partially meaningful")

# ── Main ────────────────────────────────────────────────────────────────────

def main(args):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {args.model_name}...")
    model = load_model(args.model_name, device)
    model.eval()

    print(f"Loading SAE from checkpoints/sae_layer_{args.layer}.pt ...")
    d_model = model.cfg.d_model
    d_sae   = d_model * args.expansion_factor
    
    # Load with bfloat16 to match model types and stop memory issues
    bfloat16_dtype = torch.bfloat16 if model.cfg.dtype == "bfloat16" else torch.float32
    sae = TopKSAE(d_in=d_model, d_sae=d_sae, k=args.k).to(device)
    sae.load_state_dict(
        torch.load(f"checkpoints/sae_layer_{args.layer}.pt",
                   map_location=device, weights_only=True)
    )
    sae = sae.to(dtype=bfloat16_dtype)
    sae.eval()

    # Run tests
    fve  = test_reconstruction(model, sae, args.layer, device)
    freq = test_sparsity(model, sae, args.layer, device)
    test_semantic_overlap(model, sae, args.layer, device)

    # Test 4 uses your actual top features from diff_means results
    top_hp_features = [25306, 791, 21940, 5031, 19014, 28584, 12666, 11119, 18836]
    test_ablation(model, sae, args.layer, device, top_hp_features)

    # Final summary
    section("SUMMARY")
    print(f"  Fraction variance explained : {fve:.3f}  {'✅' if fve > 0.80 else '❌'}")
    dead_pct = (freq == 0).float().mean().item()
    print(f"  Dead neuron percentage      : {dead_pct*100:.1f}%  {'✅' if dead_pct < 0.60 else '❌'}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",            type=int,   default=12)
    parser.add_argument("--model_name",       type=str,   default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--expansion_factor", type=int,   default=8)
    parser.add_argument("--k",                type=int,   default=32)
    args = parser.parse_args()
    main(args)