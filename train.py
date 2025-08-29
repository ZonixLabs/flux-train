#!/usr/bin/env python3
# train_kontext_multi_simple_fix.py
# Minimal FLUX.1-Kontext LoRA trainer (multi-context + assets, fixed config, SDPA on, safe save)

import os, glob, math
from dataclasses import dataclass
from typing import List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# New SDPA kernel context manager
try:
    from torch.nn.attention import sdpa_kernel
    _sdpa_ctx = sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    _sdpa_note = "[sdpa] flash/mem_efficient enabled"
except Exception:
    _sdpa_ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    _sdpa_note = "[sdpa] legacy sdp_kernel() used"

from flux.util import (
    load_t5,
    load_clip,
    load_ae,
    load_flow_model,
    PREFERED_KONTEXT_RESOLUTIONS,
)

# ----- LoRA targets -----

TARGET_MODULES = (
    # Double blocks - all attention and MLP
    [f"double_blocks.{i}.{part}" for i in range(19) 
     for part in ["img_attn.qkv", "img_attn.proj", "txt_attn.qkv", "txt_attn.proj",
                  "img_mlp.0", "img_mlp.2", "txt_mlp.0", "txt_mlp.2"]]
    # Single blocks - attention, linear, and projections
    + [f"single_blocks.{i}.{part}" for i in range(38)
       for part in ["attn.to_k", "attn.to_q", "attn.to_v", 
                    "linear1", "linear2", "proj_mlp", "proj_out"]]
)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# ---------------- Dataset ----------------
class KontextSeqDataset(Dataset):
    """
    Each dir:
      in/ -> context frames
      out.* -> target
      prompt.txt
      assets/ -> reference images (optional)
    """
    def __init__(self, root: str):
        item_dirs = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
        items = []
        for d in sorted(item_dirs):
            prompt = os.path.join(d, "prompt.txt")
            out_img = None
            for e in IMG_EXTS:
                cand = os.path.join(d, f"out{e}")
                if os.path.exists(cand):
                    out_img = cand
                    break
            in_dir = os.path.join(d, "in")
            if not (os.path.exists(prompt) and out_img and os.path.isdir(in_dir)):
                continue

            # context frames
            ctx = [p for p in glob.glob(os.path.join(in_dir, "*"))
                   if os.path.splitext(p)[1].lower() in IMG_EXTS]
            if not ctx and not os.path.isdir(os.path.join(d, "assets")):
                continue
            def _key(p):
                name = os.path.splitext(os.path.basename(p))[0]
                try: return (0, int(name))
                except: return (1, name)
            ctx = sorted(ctx, key=_key)

            # assets (optional)
            assets = []
            assets_dir = os.path.join(d, "assets")
            if os.path.isdir(assets_dir):
                assets = [p for p in glob.glob(os.path.join(assets_dir, "*"))
                          if os.path.splitext(p)[1].lower() in IMG_EXTS]
                assets = sorted(assets, key=_key)

            items.append((ctx, out_img, prompt, assets))
        if not items:
            raise FileNotFoundError(f"No valid items in {root}")
        self.items = items

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate_passthrough(batch):
    assert len(batch) == 1
    return batch[0]


# --------------- Helpers ----------------
def snap_resolution_from_image(img: Image.Image) -> tuple[int, int]:
    w, h = img.size
    ar = w / max(1, h)
    _, W_px, H_px = min((abs(ar - W0 / H0), W0, H0) for (W0, H0) in PREFERED_KONTEXT_RESOLUTIONS)
    W_tok = 2 * (W_px // 16)
    H_tok = 2 * (H_px // 16)
    return int(W_tok), int(H_tok)

@torch.no_grad()
def encode_image_to_tokens(ae, pil_img: Image.Image, device, W_tok: int, H_tok: int) -> torch.Tensor:
    W_tok = max(2, (W_tok // 2) * 2)
    H_tok = max(2, (H_tok // 2) * 2)
    img = pil_img.resize((8 * W_tok, 8 * H_tok), Image.Resampling.LANCZOS).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    lat = ae.encode(x)
    tok = rearrange(lat, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return tok

def make_ids(H_tok: int, W_tok: int, tau: float, bs: int, device, dtype) -> torch.Tensor:
    H2, W2 = H_tok // 2, W_tok // 2
    ids = torch.zeros(H2, W2, 3, device=device, dtype=torch.float32)
    ids[..., 0] = tau
    ids[..., 1] = torch.arange(H2, device=device)[:, None]
    ids[..., 2] = torch.arange(W2, device=device)[None, :]
    ids = rearrange(ids, "h w c -> (h w) c").unsqueeze(0).repeat(bs, 1, 1)
    return ids.to(dtype)

def sample_t_logit_normal(batch: int, device, dtype, mu=0.0, sigma=1.0, alpha: float | None = None) -> torch.Tensor:
    if alpha is not None and alpha != 1.0:
        mu = mu + math.log(alpha)
    y = torch.empty((batch,), device=device, dtype=dtype).normal_(mean=mu, std=sigma)
    t = torch.sigmoid(y)
    eps = torch.finfo(dtype).eps
    return t.clamp(eps, 1 - eps)

def load_batch_txt(t5, prompts: List[str], device):
    return t5(prompts).to(device)

def load_batch_vec(clip, prompts: List[str], device):
    return clip(prompts).to(device)

def force_lora_dtype_to(module: nn.Module, dtype: torch.dtype):
    for n, p in module.named_parameters():
        if "lora_" in n and p.dtype != dtype:
            p.data = p.data.to(dtype)

def save_lora(accelerator, model, step, cfg, prefix="step"):
    """Save just the LoRA adapter"""
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_path = os.path.join(cfg.out_dir, f"lora-{prefix}-{step}")
        unwrapped.save_pretrained(save_path, safe_serialization=True)
        print(f"Saved LoRA to {save_path}")
        return save_path
    return None


# --------------- Config ----------------
@dataclass
class TrainCfg:
    data_dir: str = os.environ.get("DATA_DIR", "data/train")
    out_dir: str  = os.environ.get("OUT_DIR", "out_kontext_lora")
    steps: int    = int(os.environ.get("STEPS", "500"))
    lr: float     = float(os.environ.get("LR", "1e-5"))
    seed: int     = int(os.environ.get("SEED", "42"))
    epochs: int   = int(os.environ.get("EPOCHS", "100"))

    max_ctx: int        = int(os.environ.get("MAX_CTX", "6"))
    ctx_downscale: int  = int(os.environ.get("CTX_DOWNSCALE", "2"))
    assets_downscale: int = int(os.environ.get("ASSETS_DOWNSCALE", "3"))
    time_spacing: float = float(os.environ.get("TIME_SPACING", "1.0"))

    debug_every: int     = int(os.environ.get("DEBUG_EVERY", "1"))
    model_name: str = os.environ.get("MODEL_NAME", "flux-dev-kontext")
    
    # Scheduler settings
    warmup_ratio: float = float(os.environ.get("WARMUP_RATIO", "0.1"))
    min_lr: float = float(os.environ.get("MIN_LR", "1e-7"))
    
    # Save settings - just one setting for LoRA saving
    save_every: int = int(os.environ.get("SAVE_EVERY", "100"))


# --------------- Main ----------------
def main():
    cfg = TrainCfg()
    accelerator = Accelerator(gradient_accumulation_steps=8)
    set_seed(cfg.seed)
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(cfg.out_dir, exist_ok=True)
        print(f"[cfg] data={cfg.data_dir} out={cfg.out_dir} steps={cfg.steps} lr={cfg.lr}")
        print(f"[cfg] K(max_ctx)={cfg.max_ctx} ctx_downscale={cfg.ctx_downscale} assets_downscale={cfg.assets_downscale}")
        print(f"[cfg] gradient_accumulation_steps={accelerator.gradient_accumulation_steps}")
        print(f"[cfg] warmup_ratio={cfg.warmup_ratio} min_lr={cfg.min_lr}")
        print(f"[cfg] save_every={cfg.save_every}")
        print(_sdpa_note)
        wandb.init(project="kontext-multi", name=f"run-seed{cfg.seed}", config=cfg.__dict__)

    device = accelerator.device

    # Load modules
    t5   = load_t5(device, max_length=512)
    clip = load_clip(device)
    ae   = load_ae(cfg.model_name, device); ae.eval()
    model = load_flow_model(cfg.model_name, device).to(device)
    model_dtype = next(model.parameters()).dtype

    # LoRA
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=TARGET_MODULES)
    model = get_peft_model(model, lcfg)
    force_lora_dtype_to(model, model_dtype)
    model.train()

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"[lora] trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Data
    ds = KontextSeqDataset(cfg.data_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True, drop_last=False,
                    num_workers=2, pin_memory=True, collate_fn=collate_passthrough)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    
    # Setup LR scheduler
    warmup_steps = int(cfg.steps * cfg.warmup_ratio)
    warmup_scheduler = LinearLR(
        opt, 
        start_factor=0.1,  # start at 10% of target LR
        total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        opt,
        T_max=cfg.steps - warmup_steps,
        eta_min=cfg.min_lr
    )
    lr_scheduler = SequentialLR(
        opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    model, opt, dl, lr_scheduler = accelerator.prepare(model, opt, dl, lr_scheduler)

    step = 0
    accumulated_loss = 0.0
    micro_step = 0  # tracks individual forward passes
    best_loss = float('inf')

    with _sdpa_ctx:
        epoch = 0
        while step < cfg.steps and epoch < cfg.epochs:
            if hasattr(dl, "sampler") and hasattr(dl.sampler, "set_epoch"):
                dl.sampler.set_epoch(epoch)
            
            for full_ctx_paths, out_path, prompt_path, asset_paths in dl:
                with accelerator.accumulate(model):
                    # target
                    tgt_img = Image.open(out_path).convert("RGB")
                    W_tok, H_tok = snap_resolution_from_image(tgt_img)
                    x_seq = encode_image_to_tokens(ae, tgt_img, device, W_tok, H_tok).to(model_dtype)
                    L_tgt, D = x_seq.shape[1], x_seq.shape[2]

                    # context frames
                    ctx_paths = full_ctx_paths[-cfg.max_ctx:] if cfg.max_ctx > 0 else full_ctx_paths
                    ctx_tokens, ctx_ids, tau_vals = [], [], []
                    for i, p in enumerate(ctx_paths):
                        cimg = Image.open(p).convert("RGB")
                        Wc, Hc = max(2, (W_tok // cfg.ctx_downscale)), max(2, (H_tok // cfg.ctx_downscale))
                        Wc, Hc = (Wc // 2) * 2, (Hc // 2) * 2
                        ctok = encode_image_to_tokens(ae, cimg, device, Wc, Hc).to(model_dtype)
                        ctx_tokens.append(ctok)
                        tau = 1.0 + i * cfg.time_spacing
                        ctx_ids.append(make_ids(Hc, Wc, tau=tau, bs=1, device=device, dtype=model_dtype))
                        tau_vals.append(tau)

                    # asset frames
                    asset_tokens, asset_ids = [], []
                    if asset_paths:
                        Wa, Ha = max(2, (W_tok // cfg.assets_downscale)), max(2, (H_tok // cfg.assets_downscale))
                        Wa, Ha = (Wa // 2) * 2, (Ha // 2) * 2
                        for j, p in enumerate(asset_paths):
                            aimg = Image.open(p).convert("RGB")
                            atok = encode_image_to_tokens(ae, aimg, device, Wa, Ha).to(model_dtype)
                            asset_tokens.append(atok)
                            tau = 101.0 + j
                            asset_ids.append(make_ids(Ha, Wa, tau=tau, bs=1, device=device, dtype=model_dtype))

                    # build sequences
                    parts = [x_seq]
                    idparts = [make_ids(H_tok, W_tok, tau=0.0, bs=1, device=device, dtype=model_dtype)]
                    if ctx_tokens:
                        parts.append(torch.cat(ctx_tokens, dim=1))
                        idparts.append(torch.cat(ctx_ids, dim=1))
                    if asset_tokens:
                        parts.append(torch.cat(asset_tokens, dim=1))
                        idparts.append(torch.cat(asset_ids, dim=1))
                    
                    img_seq = torch.cat(parts, dim=1)
                    img_ids = torch.cat(idparts, dim=1)

                    # text
                    prompt = open(prompt_path, "r", encoding="utf-8").read().strip()
                    txt = load_batch_txt(t5, [prompt], device).to(model_dtype)
                    y   = load_batch_vec(clip, [prompt], device).to(model_dtype)
                    txt_ids = torch.zeros(1, txt.shape[1], 3, device=device, dtype=model_dtype)

                    # t sample
                    H_px, W_px = 8 * H_tok, 8 * W_tok
                    alpha_now = 3.0 if max(H_px, W_px) >= 1024 else 1.0
                    t = sample_t_logit_normal(1, device=device, dtype=model_dtype, mu=0.0, sigma=1.0, alpha=alpha_now)
                    eps = torch.randn_like(x_seq, device=device, dtype=model_dtype)
                    z_t = (1.0 - t[:, None, None]) * x_seq + t[:, None, None] * eps
                    v_target = (eps - x_seq)

                    # same with noisy input
                    parts_in = [z_t]
                    idparts_in = [make_ids(H_tok, W_tok, tau=0.0, bs=1, device=device, dtype=model_dtype)]
                    if ctx_tokens:
                        parts_in.append(torch.cat(ctx_tokens, dim=1))
                        idparts_in.append(torch.cat(ctx_ids, dim=1))
                    if asset_tokens:
                        parts_in.append(torch.cat(asset_tokens, dim=1))
                        idparts_in.append(torch.cat(asset_ids, dim=1))
                    img_input = torch.cat(parts_in, dim=1)
                    img_input_ids = torch.cat(idparts_in, dim=1)

                    # forward
                    pred = model(img=img_input, img_ids=img_input_ids,
                                 txt=txt, txt_ids=txt_ids, y=y,
                                 timesteps=t.to(device=device, dtype=model_dtype),
                                 guidance = torch.full((1,), 1.0, device=device, dtype=model_dtype))
                    pred = pred[:, :L_tgt]

                    loss = F.mse_loss(pred.float(), v_target.float()).to(torch.float32)
                    accelerator.backward(loss)
                    
                    # Accumulate loss for logging
                    accumulated_loss += loss.item()
                    micro_step += 1
                    
                    # Only update weights when gradients are synchronized
                    if accelerator.sync_gradients:
                        # Gradient clipping
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        opt.step()
                        opt.zero_grad()
                        lr_scheduler.step()  # Step the scheduler
                        step += 1
                        
                        # Calculate average loss
                        avg_loss = accumulated_loss / micro_step
                        
                        # Save LoRA periodically
                        if step % cfg.save_every == 0:
                            save_lora(accelerator, model, step, cfg)
                            
                            # Save best model separately
                            if avg_loss < best_loss:
                                best_loss = avg_loss
                                save_lora(accelerator, model, step, cfg, prefix="best")
                                if is_main:
                                    print(f"New best loss: {best_loss:.4f}")
                        
                        # Log the averaged loss and current LR
                        if is_main and (step % cfg.debug_every == 0):
                            current_lr = lr_scheduler.get_last_lr()[0]
                            print(f"[step {step}/{cfg.steps}] loss={avg_loss:.4f} lr={current_lr:.2e} (avg over {micro_step} samples)")
                            wandb.log({"loss": avg_loss, "lr": current_lr, "step": step})
                            accumulated_loss = 0.0
                            micro_step = 0
                        
                        if step >= cfg.steps:
                            break
                
                if step >= cfg.steps:
                    break
            
            epoch += 1

    # Final save
    if is_main:
        save_lora(accelerator, model, step, cfg, prefix="final")
        print(f"Training complete!")
        wandb.finish()

if __name__ == "__main__":
    main()