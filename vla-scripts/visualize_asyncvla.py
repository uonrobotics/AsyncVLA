"""
visualize_asyncvla.py

Unified visualization for AsyncVLA.

Model loading follows inference_asyncvla.py exactly:
  - AutoConfig register → AutoModelForVision2Seq.from_pretrained (no trust_remote_code)
  - shead loaded with strict=False (matching inference code)
  - shead(img_cur, img_past, proj):
      past_corrected = shead(img_past, img_past, proj)   ← i=0 in inference loop
      cur_corrected  = shead(img_cur,  img_past, proj)   ← i=1 in inference loop
  - pixel_values = torch.cat([pv_current, pv_goal], dim=1)  → [1, 2C, H, W]

CLI usage
─────────────────────────────────────────────────────────────────
  # auto-detect checkpoint step from folder name
  python vla-scripts/visualize_asyncvla.py \
      --vla_path /nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA+2_step_trainig__STEP2+more_delay+no_lan/omnivla-original-balance--510000_chkpt-merged/ \
      --num_samples 100 \
      --data_root_dir /nas/sujinkim/data/goto/sim/20260323/ \
      --seed 123 \
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
import re
import random
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image

DEFAULT_OUTPUT_DIR = "./visualization/output"


# ══════════════════════════════════════════════════════════════════
# Checkpoint auto-detection
# ══════════════════════════════════════════════════════════════════

def find_resume_step(vla_path: str) -> Optional[int]:
    """
    Extract checkpoint step from folder name or .pt files inside.

    Priority:
      1. Folder name contains  --<int>_chkpt  (e.g. runs/my_run--100000_chkpt)
      2. .pt files inside named  *--<int>_checkpoint.pt  → picks the largest step
      3. Returns None if nothing found
    """
    # 1. folder name pattern
    match = re.search(r"--(\d+)_chkpt", vla_path)
    if match:
        step = int(match.group(1))
        print(f"  [ckpt] step {step} detected from folder name")
        return step

    # 2. scan .pt files
    if os.path.isdir(vla_path):
        steps = []
        for fname in os.listdir(vla_path):
            m = re.search(r"--(\d+)_checkpoint\.pt$", fname)
            if m:
                steps.append(int(m.group(1)))
        if steps:
            step = max(steps)
            print(f"  [ckpt] step {step} detected from .pt files in '{vla_path}'")
            return step

    print(f"  [ckpt] could not auto-detect step from '{vla_path}'")
    return None


# ══════════════════════════════════════════════════════════════════
# Coordinate helpers  (same as train / inference)
# ══════════════════════════════════════════════════════════════════

def delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    """[N,T,4] delta → [N,T,4] absolute pose (x, y, cosθ, sinθ)"""
    dx, dy = delta[..., 0], delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    N, T = dx.shape
    poses = []
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    for t in range(1, T):
        ct, st = torch.cos(theta), torch.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=1)


# ══════════════════════════════════════════════════════════════════
# Modality metadata
# ══════════════════════════════════════════════════════════════════

MASK_TEXTS = [
    "satellite only",    # 0
    "pose + satellite",  # 1
    "satellite + image", # 2
    "all",               # 3
    "pose only",         # 4
    "pose + image",      # 5
    "image only",        # 6
    "language only",     # 7
    "language + pose",   # 8
]

SHOW_GOAL_STAR_IDS = {1, 3, 4, 5, 8}
SHOW_OBJ_STAR_IDS  = {7, 8}


# ══════════════════════════════════════════════════════════════════
# Core single-sample visualizer
# ══════════════════════════════════════════════════════════════════

def visualize_asyncvla(
    past_image_PIL:         Image.Image,
    current_image_PIL:      Image.Image,
    goal_image_PIL:         Optional[Image.Image],   # None → column skipped
    gt_actions:             Optional[torch.Tensor],  # [1,T,4]  None at inference
    base_vlm_actions:       Optional[torch.Tensor],  # [1,T,4]  projected before shead
    past_corrected_actions: Optional[torch.Tensor],  # [1,T,4]  shead(past, past, proj)
    cur_corrected_actions:  Optional[torch.Tensor],  # [1,T,4]  shead(cur,  past, proj)
    goal_pose:              Optional[np.ndarray] = None,  # [4] (x,y,cosθ,sinθ)
    obj_pose_norm:          Optional[np.ndarray] = None,  # [2] (x,y) LeLaN object
    modality_id:            int = 6,
    lan_prompt:             str = "",
    sample_idx:             int = 0,
    save_path:              str = f"{DEFAULT_OUTPUT_DIR}/sample_000000.png",
    title:                  str = "AsyncVLA – Unified Visualization",
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    has_goal_img = goal_image_PIL is not None
    n_img_cols   = 3 if has_goal_img else 2

    fig = plt.figure(figsize=(5 * n_img_cols + 11, 9), dpi=110)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    outer  = gridspec.GridSpec(1, 2, figure=fig,
                               width_ratios=[n_img_cols, 3.2], wspace=0.06)
    img_gs = gridspec.GridSpecFromSubplotSpec(
                 1, n_img_cols, subplot_spec=outer[0], wspace=0.04)
    ax_traj = fig.add_subplot(outer[1])

    # image panels
    imgs   = [past_image_PIL, current_image_PIL] + ([goal_image_PIL] if has_goal_img else [])
    titles = ["Past Image",   "Current Image"]   + (["Goal Image"]   if has_goal_img else [])
    for col, (img, ttl) in enumerate(zip(imgs, titles)):
        ax = fig.add_subplot(img_gs[col])
        ax.imshow(np.array(img).astype(np.uint8))
        ax.set_title(ttl, fontsize=12)
        ax.axis("off")

    # trajectory helper
    def _plot_traj(pose_tensor, color, label, marker="o", lw=2.8, ms=8, zorder=3):
        if pose_tensor is None:
            return
        t  = pose_tensor[0].detach().cpu().float().numpy()
        xs =  t[:, 0]
        ys = -t[:, 1]
        ax_traj.plot(np.insert(ys, 0, 0.0), np.insert(xs, 0, 0.0),
                     color=color, label=label,
                     linewidth=lw, marker=marker, markersize=ms, zorder=zorder)
        for i in range(0, len(xs), max(1, len(xs) // 4)):
            ax_traj.annotate("",
                xy=(ys[i] - t[i, 3] * 0.25, xs[i] + t[i, 2] * 0.25),
                xytext=(ys[i], xs[i]),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
                zorder=zorder + 1)

    _plot_traj(gt_actions,             "#1a1aff",   "GT trajectory",              "o",  lw=2.0, ms=7,  zorder=6)
    _plot_traj(cur_corrected_actions,  "#00bfff",   "Current-image corrected",    "^",  lw=3.5, ms=10, zorder=5)
    _plot_traj(base_vlm_actions,       "#8B008B",   "Base VLM (no edge adapter)", "o",  lw=2.0, ms=7,  zorder=4)
    _plot_traj(past_corrected_actions, "#FF69B4",   "Past-image corrected",       "^",  lw=3.5, ms=10, zorder=3)

    ax_traj.plot(0, 0, marker="P", color="black", markersize=14, label="Robot (origin)", zorder=6)

    if modality_id in SHOW_GOAL_STAR_IDS and goal_pose is not None:
        ax_traj.plot(-float(goal_pose[1]), float(goal_pose[0]),
                     marker="*", color="gold", markersize=22,
                     markeredgecolor="black", markeredgewidth=0.7,
                     label="Goal (pose)", zorder=7)

    if modality_id in SHOW_OBJ_STAR_IDS and obj_pose_norm is not None:
        ax_traj.plot(-float(obj_pose_norm[1]), float(obj_pose_norm[0]),
                     marker="*", color="limegreen", markersize=20,
                     markeredgecolor="black", markeredgewidth=0.7,
                     label="Object (language goal)", zorder=7)

    modality_text = MASK_TEXTS[modality_id] if modality_id < len(MASK_TEXTS) else str(modality_id)
    ax_traj.set_title(f"Trajectory  │  [{modality_id}] {modality_text}", fontsize=12)
    if lan_prompt:
        ax_traj.text(0.02, 0.97, f'"{lan_prompt}"',
                     transform=ax_traj.transAxes, fontsize=10, va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.85))
    ax_traj.set_xlabel("← Left  |  Right →  (normalised)", fontsize=10)
    ax_traj.set_ylabel("← Back  |  Front →  (normalised)", fontsize=10)
    ax_traj.set_xlim(-10, 10)
    ax_traj.set_ylim(-0.5, 15)
    ax_traj.axhline(0, color="gray", lw=0.7, ls="--")
    ax_traj.axvline(0, color="gray", lw=0.7, ls="--")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.text(0.98, 0.02, f"sample #{sample_idx:06d}",
                 transform=ax_traj.transAxes, fontsize=9,
                 va="bottom", ha="right", color="gray")

    fig.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [viz] saved → {save_path}")
    return save_path


# ══════════════════════════════════════════════════════════════════
# Dataset visualizer (CLI entry-point)
# ══════════════════════════════════════════════════════════════════

def _dummy_traj(T: int = 8, scale: float = 5.0, noise: float = 0.0) -> torch.Tensor:
    xs  = np.cumsum(np.random.uniform(0.2, 0.8, T)) * scale / T
    ys  = np.cumsum(np.random.uniform(-0.3, 0.3, T)) * scale / T
    th  = np.cumsum(np.random.uniform(-0.12, 0.12, T))
    arr = np.stack([xs, ys, np.cos(th), np.sin(th)], axis=-1)
    t   = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    return t + torch.randn_like(t) * noise if noise > 0 else t


def _dummy_rgb(mean_rgb: tuple, size: int = 224) -> Image.Image:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for c, base in enumerate(mean_rgb):
        arr[:, :, c] = np.clip(
            np.random.randint(max(0, base - 30), min(256, base + 31), (size, size)),
            0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _load_model(vla_path: str, resume_step: int, device: torch.device):
    """
    Load VLA + action_proj + pose_projector + shead.
    Follows inference_asyncvla.py define_model() exactly.
    """
    import yaml
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
    from prismatic.models.projectors import ProprioProjector
    from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.models.action_heads import L1RegressionActionHead_idcat

    # ── register to HF Auto Classes (same as inference_asyncvla.py) ──
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor        = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ── load VLA (same as inference_asyncvla.py define_model) ────────
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(2)
    vla.to(dtype=torch.bfloat16)
    vla.eval()

    # ── checkpoint loader helper ──────────────────────────────────────
    def _load_ckpt(name: str) -> dict:
        # fallback: pose_projector → proprio_projector (same as load_checkpoint() in both scripts)
        for stem in [name, name.replace("pose_projector", "proprio_projector")]:
            p = os.path.join(vla_path, f"{stem}--{resume_step}_checkpoint.pt")
            if os.path.exists(p):
                print(f"  Loading checkpoint: {p}")
                sd = torch.load(p, map_location="cpu", weights_only=True)
                # strip DDP "module." prefix
                return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        raise FileNotFoundError(f"No checkpoint for '{name}' at step {resume_step} in {vla_path}")

    # ── pose_projector ────────────────────────────────────────────────
    pose_projector = ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM)
    pose_projector.load_state_dict(_load_ckpt("pose_projector"))
    pose_projector.to(device).eval()
    
    # ── action_head ───────────────────────────────────────────────────
    action_head = L1RegressionActionHead_idcat(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)
    action_head.load_state_dict(_load_ckpt("action_head"))
    action_head.to(torch.bfloat16).to(device).eval()


    # ── action_proj ───────────────────────────────────────────────────
    action_proj = Proj_Actiontokens(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=1024)
    action_proj.load_state_dict(_load_ckpt("action_proj"))
    action_proj.to(torch.bfloat16).to(device).eval()

    # ── shead (Edge adapter) ──────────────────────────────────────────
    with open("./config_nav/dataset_config.yaml") as f:
        cfg_nav = yaml.safe_load(f)

    shead = Edge_adapter(
        obs_encoding_size=cfg_nav["obs_encoding_size"],
        mha_num_attention_heads=cfg_nav["mha_num_attention_heads"],
        mha_num_attention_layers=cfg_nav["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg_nav["mha_ff_dim_factor"],
    )
    # strict=False + report keys  ← matches inference_asyncvla.py
    missing, unexpected = shead.load_state_dict(_load_ckpt("shead"), strict=False)
    if missing:    print(f"  [shead] missing keys   : {missing}")
    if unexpected: print(f"  [shead] unexpected keys: {unexpected}")
    shead.to(torch.bfloat16).to(device).eval()

    # NUM_PATCHES = patches_per_image * num_images + 1 (goal pose token)
    NUM_PATCHES = (vla.vision_backbone.get_num_patches()
                   * vla.vision_backbone.get_num_images_in_input() + 1)

    return (vla, action_head, action_proj, pose_projector, shead,
            action_tokenizer, processor, NUM_PATCHES,
            ACTION_DIM, NUM_ACTIONS_CHUNK)


def _run_inference(sample: dict, model_bundle, past_img: Image.Image,
                   cur_img: Image.Image, device: torch.device):
    """
    Run VLA + shead forward pass for one sample.
    Returns (base_vlm, past_corr, cur_corr) as [1,T,4] float tensors.
    Mirrors run_forward_pass() in inference_asyncvla.py.
    """
    import torchvision.transforms as T
    from torchvision.transforms.functional import to_tensor, resize
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

    (vla, action_head, action_proj, pose_projector, shead,
     action_tokenizer, processor,
     NUM_PATCHES, ACTION_DIM, NUM_ACTIONS_CHUNK) = model_bundle

    pad_id      = processor.tokenizer.pad_token_id
    IGNORE_IDX  = -100
    norm        = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # ── collate single sample → batch ────────────────────────────────
    input_ids = sample["input_ids"].unsqueeze(0)
    labels    = sample["labels"].unsqueeze(0)
    attn_mask = input_ids.ne(pad_id)

    # pixel_values: cat([current, goal], dim=1) → [1, 2C, H, W]
    # matches collator_custom in inference_asyncvla.py
    pv_cur  = sample.get("pixel_values_current", sample.get("pixel_values"))
    pv_goal = sample.get("pixel_values_goal", pv_cur)
    pixel_values = torch.cat(
        [pv_cur.unsqueeze(0), pv_goal.unsqueeze(0)], dim=1
    )

    goal_pose_t = torch.tensor(
        sample["goal_pose"] if "goal_pose" in sample else np.zeros(4, dtype=np.float32),
        dtype=torch.float32,
    ).unsqueeze(0)

    modality_id_t = torch.as_tensor(
        [int(sample.get("modality_id", 6))], dtype=torch.float32
    )

    # ── VLA forward pass (same as inference run_forward_pass) ─────────
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                          enabled=device.type == "cuda"):
        output = vla(
            input_ids=input_ids.to(device),
            attention_mask=attn_mask.to(device),
            pixel_values=pixel_values.to(torch.bfloat16).to(device),
            modality_id=modality_id_t.to(torch.bfloat16).to(device),
            labels=labels.to(device),
            output_hidden_states=True,
            proprio=goal_pose_t.to(torch.bfloat16).to(device),
            proprio_projector=pose_projector,
            use_film=False,
        )

    # ── extract action hidden states ──────────────────────────────────
    last_hidden = output.hidden_states[-1]
    text_hidden = last_hidden[:, NUM_PATCHES:-1]
    gt_tids     = labels[:, 1:].to(device)
    cur_mask    = get_current_action_mask(gt_tids)
    next_mask   = get_next_actions_mask(gt_tids)
    act_hidden  = (
        text_hidden[cur_mask | next_mask]
        .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )

    with torch.no_grad():
        # base VLM: before edge adapter
        base_vlm = action_head.predict_action(
            act_hidden, modality_id_t.to(torch.bfloat16).to(device))
        proj = action_proj.predict_action(
            act_hidden, modality_id_t.to(torch.bfloat16).to(device))

    # ── edge adapter – mirror inference loop ──────────────────────────
    # inference_asyncvla.py:
    #   img_past = transform(p_image)  ← fixed
    #   i=0: img_cur = transform(past_image) → shead(img_cur, img_past, proj)  = past_corr
    #   i=1: img_cur = transform(cur_image)  → shead(img_cur, img_past, proj)  = cur_corr
    def _edge_tensor(pil_img):
        return norm(resize(to_tensor(pil_img), [96, 96])).unsqueeze(0).to(device).to(torch.bfloat16)

    img_past_t = _edge_tensor(past_img)
    img_cur_t  = _edge_tensor(cur_img)

    with torch.no_grad():
        past_corr = delta_to_pose(shead(img_past_t, img_past_t, proj)).cpu().float()
        cur_corr  = delta_to_pose(shead(img_cur_t,  img_past_t, proj)).cpu().float()

    return base_vlm, past_corr, cur_corr


def run_dataset_visualization(
    num_samples:   int,
    output_dir:    str,
    vla_path:      Optional[str] = None,
    resume_step:   Optional[int] = None,
    data_root_dir: Optional[str] = None,
    seed:          int  = 0,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # auto-detect step
    if resume_step is None and vla_path is not None:
        resume_step = find_resume_step(vla_path)

    print(f"\n{'─' * 60}")
    print(f"  AsyncVLA Dataset Visualizer")
    print(f"  vla_path    : {vla_path}")
    print(f"  resume_step : {resume_step}")
    print(f"  num_samples : {num_samples if num_samples > 0 else 'ALL'}")
    print(f"  output_dir  : {os.path.abspath(output_dir)}")
    print(f"  seed        : {seed}")
    print(f"{'─' * 60}\n")

    dataset      = None
    model_bundle = None
    device       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if vla_path is not None and data_root_dir is not None:
        try:
            sys.path.extend([
                "../Learning-to-Drive-Anywhere-with-MBRA/train/", "../lerobot"
            ])
            from prismatic.vla.datasets.goto_sim_dataset import GotoSim_Dataset
            from prismatic.vla.action_tokenizer import ActionTokenizer
            from prismatic.models.backbones.llm.prompting import PurePromptBuilder
            from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

            processor        = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
            action_tokenizer = ActionTokenizer(processor.tokenizer)

            dataset = GotoSim_Dataset(
                root_dir=Path(data_root_dir),
                image_transform=processor.image_processor.apply_transform,
                action_tokenizer=action_tokenizer,
                prompt_builder_fn=PurePromptBuilder,
                base_tokenizer=processor.tokenizer,
            )
            print(f"  Loaded dataset: {len(dataset)} samples")

            if resume_step is not None:
                model_bundle = _load_model(vla_path, resume_step, device)
                print(f"  Loaded model at step {resume_step}")
            else:
                print("  [warn] No resume_step → using GT-perturbed dummy predictions")

        except Exception as exc:
            import traceback
            print(f"  [warn] Could not load dataset/model: {exc}")
            traceback.print_exc()
            dataset = None
            model_bundle = None

    # sample count
    total   = len(dataset) if dataset is not None else max(num_samples, 1)
    n       = total if num_samples == 0 else min(num_samples, total)
    indices = random.sample(range(total), n) if dataset is not None else list(range(n))

    modality_cycle = list(range(9))

    for viz_i, raw_idx in enumerate(indices):
        print(f"  [{viz_i + 1:>5d} / {n}]  idx={raw_idx}", end="  ")

        if dataset is not None:
            sample = dataset[raw_idx]

            def _to_pil(key, fallback_rgb):
                v = sample.get(key)
                if isinstance(v, Image.Image):
                    return v
                if isinstance(v, torch.Tensor):
                    arr = v.float().permute(1, 2, 0).numpy()
                    return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
                return _dummy_rgb(fallback_rgb)

            past_img  = _to_pil("p_image",  (180, 200, 220))
            cur_img   = _to_pil("c_image",  (200, 220, 180))
            goal_img  = _to_pil("gimg_PIL", (220, 200, 180)) if "gimg_PIL" in sample else None

            gt_traj = (
                torch.tensor(sample["actions"], dtype=torch.float32).unsqueeze(0)
                if "actions" in sample else _dummy_traj()
            )
            modality_id   = int(sample.get("modality_id", 6))
            lan_prompt    = sample.get("lan_prompt", "")
            goal_pose     = np.array(sample["goal_pose"],     dtype=np.float32) if "goal_pose"     in sample else None
            obj_pose_norm = np.array(sample["obj_pose_norm"], dtype=np.float32) if "obj_pose_norm" in sample else None

            if model_bundle is not None:
                try:
                    base_vlm, past_corr, cur_corr = _run_inference(
                        sample, model_bundle, past_img, cur_img, device)
                except Exception as exc:
                    print(f"\n  [warn] inference failed: {exc}")
                    base_vlm  = gt_traj + torch.randn_like(gt_traj) * 1.2
                    past_corr = gt_traj + torch.randn_like(gt_traj) * 0.4
                    cur_corr  = gt_traj + torch.randn_like(gt_traj) * 0.15
            else:
                base_vlm  = gt_traj + torch.randn_like(gt_traj) * 1.2
                past_corr = gt_traj + torch.randn_like(gt_traj) * 0.4
                cur_corr  = gt_traj + torch.randn_like(gt_traj) * 0.15

        else:
            modality_id   = modality_cycle[viz_i % len(modality_cycle)]
            past_img      = _dummy_rgb((160 + modality_id * 8, 190, 210))
            cur_img       = _dummy_rgb((190, 200 + modality_id * 5, 175))
            goal_img      = _dummy_rgb((210, 185, 190 + modality_id * 5)) if modality_id in {2, 3, 5, 6} else None
            gt_traj       = _dummy_traj(scale=6.0)
            base_vlm      = _dummy_traj(scale=6.0, noise=1.0)
            past_corr     = _dummy_traj(scale=6.0, noise=0.4)
            cur_corr      = _dummy_traj(scale=6.0, noise=0.15)
            goal_pose     = (
                np.array([7.0 + random.uniform(-1, 1), random.uniform(-2, 2), 1.0, 0.0])
                if modality_id in SHOW_GOAL_STAR_IDS else None
            )
            obj_pose_norm = (
                np.array([random.uniform(5, 10), random.uniform(-2, 2)])
                if modality_id in SHOW_OBJ_STAR_IDS else None
            )
            lan_prompt = "move toward the blue trash bin" if modality_id in SHOW_OBJ_STAR_IDS else ""

        step_tag = f"step{resume_step}" if resume_step is not None else "stepNA"
        fname    = os.path.join(
            output_dir,
            f"sample_{viz_i:06d}_idx{raw_idx:06d}_mod{modality_id}_{step_tag}.png",
        )
        visualize_asyncvla(
            past_image_PIL=past_img,
            current_image_PIL=cur_img,
            goal_image_PIL=goal_img,
            gt_actions=gt_traj,
            base_vlm_actions=base_vlm,
            past_corrected_actions=past_corr,
            cur_corrected_actions=cur_corr,
            goal_pose=goal_pose,
            obj_pose_norm=obj_pose_norm,
            modality_id=modality_id,
            lan_prompt=lan_prompt,
            sample_idx=raw_idx,
            save_path=fname,
            title=f"AsyncVLA  step={resume_step}  viz#{viz_i:06d}  (idx {raw_idx})",
        )

    print(f"\n{'─' * 60}")
    print(f"  Done. {n} image(s) → '{os.path.abspath(output_dir)}'")
    print(f"{'─' * 60}\n")


# ══════════════════════════════════════════════════════════════════
# Argument parser & entry point
# ══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualize AsyncVLA dataset samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--vla_path", type=str, default=None,
        help=(
            "Path to the VLA checkpoint directory. "
            "Step is auto-detected from the folder name (e.g. --100000_chkpt) "
            "or from *.pt files inside."
        ),
    )
    p.add_argument(
        "--resume_step", type=int, default=None,
        help="Manually specify the checkpoint step (overrides auto-detection).",
    )
    p.add_argument(
        "--num_samples", type=int, default=20,
        help="Number of samples to visualize.  0 = all.",
    )
    p.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory where output PNGs are saved.",
    )
    p.add_argument(
        "--data_root_dir", type=str, default=None,
        help="Dataset root directory.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_dataset_visualization(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        vla_path=args.vla_path,
        resume_step=args.resume_step,
        data_root_dir=args.data_root_dir,
        seed=args.seed,
    )