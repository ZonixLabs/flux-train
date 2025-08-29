#!/usr/bin/env python3
# merge_flux_lora.py
# Merge a PEFT LoRA adapter into a FLUX.1-Kontext base .safetensors
# - Safe fp32 matmul, preserve base dtype
# - Supports HF repo id OR local model file/dir
# - Handles PEFT key variants (with/without ".default", "base_model.model." prefix)
#
# Usage examples:
#   python merge_flux_lora.py \
#     --model black-forest-labs/FLUX.1-Kontext-dev \
#     --lora out_kontext_lora \
#     --output flux1-kontext-merged.safetensors \
#     --multiplier 0.5
#
#   python merge_flux_lora.py \
#     --model ./flux/checkpoints/black-forest-labs_FLUX.1-Kontext-dev \
#     --lora out_kontext_lora \
#     --output flux1-kontext-merged.safetensors

import argparse
import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import safetensors.torch as st

try:
    from huggingface_hub import hf_hub_download, snapshot_download  # optional
    HF_OK = True
except Exception:
    HF_OK = False


def _find_model_file(model_source: str) -> str:
    """
    Return path to a single base .safetensors file for FLUX model weights.
    Accepts:
      - local file (.safetensors)
      - local dir (search recursively)
      - HF repo id (requires huggingface_hub)
    """
    # Local file
    if os.path.isfile(model_source):
        if model_source.endswith(".safetensors"):
            print(f"[info] Using local model file: {model_source}")
            return model_source
        raise ValueError(f"Expected a .safetensors file, got: {model_source}")

    # Local dir
    if os.path.isdir(model_source):
        print(f"[info] Searching for model file in dir: {model_source}")
        cands = []
        for root, _, files in os.walk(model_source):
            for f in files:
                lf = f.lower()
                if lf.endswith(".safetensors") and "ae" not in lf:
                    cands.append(os.path.join(root, f))
        if not cands:
            raise FileNotFoundError(f"No non-AE .safetensors files under {model_source}")
        # Prefer FLUX-ish names
        pref = ["flux1-kontext", "flux", "model", "diffusion", "pytorch_model"]
        for p in pref:
            for c in cands:
                if p in os.path.basename(c).lower():
                    print(f"[found] {c}")
                    return c
        print(f"[found] {cands[0]}")
        return cands[0]

    # HF repo id
    if not HF_OK:
        raise ValueError("huggingface_hub not available; install it or pass a local path.")
    print(f"[info] Looking in HF repo: {model_source}")
    names = [
        "flux1-kontext-dev.safetensors",
        "flux1-kontext-merged.safetensors",
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "pytorch_model.safetensors",
    ]
    for n in names:
        try:
            p = hf_hub_download(repo_id=model_source, filename=n)
            print(f"[found] {n}")
            return p
        except Exception:
            pass
    # fallback: snapshot and search
    repo_path = snapshot_download(repo_id=model_source)
    return _find_model_file(repo_path)


def _load_lora(adapter_dir: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    Load PEFT adapter weights + config from a directory
    """
    adir = Path(adapter_dir)
    cfg_path = adir / "adapter_config.json"
    wt_path  = adir / "adapter_model.safetensors"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    if not wt_path.exists():
        raise FileNotFoundError(f"Missing {wt_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    wts = st.load_file(str(wt_path))
    return wts, cfg


def _normalize_base_key_from_lora_key(lora_key: str) -> Optional[str]:
    """
    Convert PEFT LoRA key -> base weight key
    Accepts patterns like:
      base_model.model.double_blocks.0.img_attn.proj.lora_A.weight
      base_model.model.double_blocks.0.img_attn.proj.lora_A.default.weight
      double_blocks.0.img_attn.proj.lora_A.weight
      double_blocks.0.img_attn.proj.lora_A.default.weight
    Returns:
      double_blocks.0.img_attn.proj.weight
    """
    k = lora_key
    if k.endswith(".default.weight"):
        k = k.replace(".default.weight", ".weight")
    if k.endswith(".lora_A.weight"):
        k = k.replace(".lora_A.weight", ".weight")
    # strip the common prefix PEFT adds
    k = k.replace("base_model.model.", "")
    # sanity
    if ".weight" not in k:
        return None
    return k


def _pair_lora_keys(lora_state: Dict[str, torch.Tensor]) -> List[Tuple[str, str]]:
    """
    Find matching (A, B) keys in the LoRA state dict.
    Returns list of tuples: (lora_A_key, lora_B_key)
    """
    a_keys = [k for k in lora_state.keys() if ".lora_A" in k and k.endswith("weight")]
    pairs = []
    for a in a_keys:
        b = a.replace(".lora_A", ".lora_B")
        if b in lora_state:
            pairs.append((a, b))
    return pairs


def merge_lora(
    base_path: str,
    lora_dir: str,
    output_path: str,
    multiplier: float = 1.0,
    dry_run: bool = False,
    print_deltas: int = 0,
) -> None:
    """
    Merge PEFT LoRA into base FLUX weights and save .safetensors
    - multiplier < 1.0 reduces update magnitude (good for sanity)
    - print_deltas > 0 prints mean|Δ| for that many merged layers
    """
    print(f"[load] base: {base_path}")
    base_sd = st.load_file(base_path)

    lora_sd, lora_cfg = _load_lora(lora_dir)
    r = lora_cfg.get("r", None)
    alpha = lora_cfg.get("lora_alpha", None)
    scaling = (alpha / r) if (r and alpha) else 1.0

    print(f"[info] base tensors: {len(base_sd)}")
    print(f"[info] lora tensors: {len(lora_sd)}")
    print(f"[info] LoRA r={r} alpha={alpha} scaling={scaling} multiplier={multiplier}")

    pairs = _pair_lora_keys(lora_sd)
    if not pairs:
        raise ValueError("No LoRA (A,B) pairs found in adapter.")

    # Determine base dtype
    any_k = next(iter(base_sd.keys()))
    base_dtype = base_sd[any_k].dtype
    print(f"[info] base dtype: {base_dtype}")

    merged = 0
    merged_list = []
    skipped = []

    # Copy base weights (we’ll mutate this dict)
    out_sd = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in base_sd.items()}

    for a_key, b_key in pairs:
        base_key = _normalize_base_key_from_lora_key(a_key)
        if base_key is None or base_key not in out_sd:
            skipped.append(a_key)
            continue

        W = out_sd[base_key]
        A = lora_sd[a_key]
        B = lora_sd[b_key]

        # LoRA update: W_new = W + (B @ A) * scaling * multiplier
        # Compute in fp32 for stability, then cast back to base dtype
        with torch.no_grad():
            Wf = W.float()
            upd = (B.float() @ A.float()) * float(scaling) * float(multiplier)
            W_new = Wf + upd
            out_sd[base_key] = W_new.to(base_dtype)

        merged += 1
        merged_list.append(base_key)

    print(f"[summary] merged layers: {merged}, skipped: {len(skipped)}")
    if skipped:
        print("  (skipped examples):")
        for s in skipped[:8]:
            print(f"   - {s}")

    if dry_run:
        print("[dry-run] Not saving output.")
        return

    # Optional post-merge delta check
    if print_deltas > 0:
        print("[check] mean|Δ| for first merged layers:")
        for k in merged_list[:print_deltas]:
            try:
                d = (out_sd[k].float() - base_sd[k].float()).abs().mean().item()
                print(f"  {k} mean|Δ| = {d:.6g}")
            except Exception:
                pass

    print(f"[save] {output_path}")
    st.save_file(out_sd, output_path)
    sz_gb = os.path.getsize(output_path) / (1024**3)
    print(f"[info] size: {sz_gb:.2f} GB")
    print("[done]")


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into FLUX.1-Kontext weights")
    ap.add_argument("--model", required=True,
                    help="Base model: HF repo id OR local .safetensors path OR local dir containing model")
    ap.add_argument("--lora", required=True,
                    help="Path to PEFT LoRA adapter folder (adapter_model.safetensors + adapter_config.json)")
    ap.add_argument("--output", "-o", default="flux1-kontext-merged.safetensors",
                    help="Output .safetensors")
    ap.add_argument("--multiplier", type=float, default=1.0,
                    help="Extra scale on LoRA update (e.g. 0.25, 0.5, 1.0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Do everything except write the file")
    ap.add_argument("--print-deltas", type=int, default=4,
                    help="Print mean|Δ| for first N merged layers")
    args = ap.parse_args()

    base_path = _find_model_file(args.model)
    merge_lora(
        base_path=base_path,
        lora_dir=args.lora,
        output_path=args.output,
        multiplier=args.multiplier,
        dry_run=args.dry_run,
        print_deltas=args.print_deltas,
    )


if __name__ == "__main__":
    main()
