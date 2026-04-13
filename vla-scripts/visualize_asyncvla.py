"""
visualize_asyncvla.py

Standalone visualization script for AsyncVLA.
Run this separately from training to visualize model predictions.

Usage:
    WORLD_SIZE=1 python visualize_asyncvla.py \
        --vla_path /path/to/checkpoint \
        --data_root_dir /path/to/dataset \
        --num_samples 150
"""

import re
import sys
from pathlib import Path

sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/", '../lerobot'
])

import os
import math
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from typing import Dict, Optional, Tuple, Type
from dataclasses import dataclass
from torchvision import transforms

import draccus
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Nav_MMN
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, IGNORE_INDEX
from prismatic.vla.datasets.goto_sim_dataset import GotoSim_Dataset
from prismatic.models.small_head import Edge_adapter, Proj_Actiontokens

import torch.distributed as dist

os.environ["TOKENIZERS_PARALLELISM"] = "false"

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


@dataclass
class VisConfig:
    vla_path: str = "./AsyncVLA_release"
    data_root_dir: Path = Path("datasets/goto_sim")
    dataset_name: str = "goto/sim"
    output_dir: str = "./visualization_standalone"
    num_samples: int = 20
    batch_size: int = 1
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 128
    lora_dropout: float = 0.0
    inference_seed: Optional[int] = 42


def remove_ddp_in_checkpoint(state_dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt")) and module_name == "pose_projector":
        module_name = "proprio_projector"
    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused,
               gradient_as_bucket_view=True)


def delta_to_pose(delta):
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    N, T = dx.shape
    poses = []
    x = dx[:, 0]
    y = dy[:, 0]
    theta = dtheta[:, 0]
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    for t in range(1, T):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=1)


def to_numpy(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy()


def to_imshow_image(x):
    if isinstance(x, Image.Image):
        return np.array(x).astype(np.uint8)
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if x.ndim == 3 and x.shape[0] not in [96, 224] and x.shape[0] >= 3:
            x = x[:3].permute(1, 2, 0)
        x = x.to(torch.float32).numpy()
        if x.max() <= 1.0:
            x = x * 255.0
        return np.clip(x, 0, 255).astype(np.uint8)
    x = np.array(x)
    if x.ndim == 3 and x.shape[0] >= 3 and x.shape[-1] not in [3, 4]:
        x = np.transpose(x[:3], (1, 2, 0))
    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def merge_batches_padding(batch_list, pad_token_id, model_max_length):
    merged = {}
    keys = batch_list[0].keys()
    for key in keys:
        values = [batch[key] for batch in batch_list]
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(first_value, list):
            combined = []
            for v in values:
                combined.extend(v)
            merged[key] = combined
    input_ids = pad_sequence(merged["input_ids"], batch_first=True, padding_value=pad_token_id)
    merged["input_ids"] = input_ids[:, :model_max_length]
    labels = pad_sequence(merged["labels"], batch_first=True, padding_value=IGNORE_INDEX)
    merged["labels"] = labels[:, :model_max_length]
    merged["attention_mask"] = merged["input_ids"].ne(pad_token_id)
    merged["attention_mask_label"] = merged["labels"].ne(IGNORE_INDEX)
    merged["goal_mask_select"] = torch.tensor(merged["modality_id"])
    return merged


def visualize_sample(
    batch_past_img, batch_curr_img, batch_goal_PIL,
    goal_pos_lan, goal_pos,
    traj_gt, traj_vlm_base, traj_edge_past, traj_edge_curr,
    goal_mask_select, lan_prompts,
    save_path,
):
    mask_type = int(goal_mask_select.item())

    fig = plt.figure(figsize=(18.5, 10.5), dpi=80)
    gs = fig.add_gridspec(3, 2)
    ax_graph = fig.add_subplot(gs[0:3, 1:2])
    ax_past_obs = fig.add_subplot(gs[0:1, 0:1])
    ax_curr_obs = fig.add_subplot(gs[1:2, 0:1])
    ax_goal = fig.add_subplot(gs[2:3, 0:1])

    ax_past_obs.imshow(to_imshow_image(batch_past_img))
    ax_curr_obs.imshow(to_imshow_image(batch_curr_img))
    ax_goal.imshow(np.array(batch_goal_PIL).astype(np.uint8))

    # GT trajectory
    x_gt = traj_gt[:, 0].numpy()
    y_gt = traj_gt[:, 1].numpy()
    ax_graph.plot(-np.insert(y_gt, 0, 0.0), np.insert(x_gt, 0, 0.0),
                  marker='o', color='red', linewidth=1.2, markersize=3, label="gt")

    # VLM raw trajectory
    if traj_vlm_base is not None:
        x_vlm = traj_vlm_base[:, 0].numpy()
        y_vlm = traj_vlm_base[:, 1].numpy()
        ax_graph.plot(-np.insert(y_vlm, 0, 0.0), np.insert(x_vlm, 0, 0.0),
                      marker='o', color='blue', linewidth=4, markersize=8, label="vlm_raw")

    # Edge past
    x_ep = traj_edge_past[:, 0].numpy()
    y_ep = traj_edge_past[:, 1].numpy()
    ax_graph.plot(-np.insert(y_ep, 0, 0.0), np.insert(x_ep, 0, 0.0),
                  marker='o', color='purple', linewidth=1.5, markersize=5, label="edge_past")

    # Edge current
    x_ec = traj_edge_curr[:, 0].numpy()
    y_ec = traj_edge_curr[:, 1].numpy()
    ax_graph.plot(-np.insert(y_ec, 0, 0.0), np.insert(x_ec, 0, 0.0),
                  marker='o', color='pink', linewidth=1.8, markersize=6, label="edge_current")

    # Goal marker - modality에 따라 조건부 표시
    if mask_type in [4, 5, 8]:
        xgt_pos = to_numpy(goal_pos[0])
        ygt_pos = to_numpy(goal_pos[1])
        ax_graph.plot(-ygt_pos, xgt_pos, marker='*', color='black', markersize=16, label="goal")
    elif mask_type == 7:
        xgt_pos = to_numpy(goal_pos_lan[0])
        ygt_pos = to_numpy(goal_pos_lan[1])
        ax_graph.plot(-ygt_pos, xgt_pos, marker='*', color='black', markersize=16, label="goal")
    # modality 6 (image only): goal 표시 안 함

    mask_texts = [
        "satellite only", "pose and satellite", "satellite and image", "all",
        "pose only", "pose and image", "image only", "language only", "language and pose"
    ]
    if mask_type < len(mask_texts):
        ax_graph.annotate(mask_texts[mask_type], xy=(-8.0, 0.5), xytext=(-20, 20),
                          fontsize=12, textcoords='offset points')
    if mask_type in [7, 8] and lan_prompts:
        ax_graph.annotate(lan_prompts, xy=(-8.0, 0.0), xytext=(-20, 20),
                          fontsize=12, textcoords='offset points')

    ax_graph.set_title("GT vs VLM vs Edge-past vs Edge-current trajectory")
    ax_graph.set_xlim(-10.0, 10.0)
    ax_graph.set_ylim(-0.1, 15.0)
    ax_graph.legend(loc='best')
    ax_past_obs.set_title("Egocentric past image", fontsize=12)
    ax_curr_obs.set_title("Egocentric current image", fontsize=12)
    ax_goal.set_title("Egocentric goal image", fontsize=12)

    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


@draccus.wrap()
def main(cfg: VisConfig) -> None:
    if cfg.inference_seed is not None:
        random.seed(cfg.inference_seed)
        np.random.seed(cfg.inference_seed)
        torch.manual_seed(cfg.inference_seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(device_id)

    with open("./config_nav/dataset_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # ── VLA path & resume step ──────────────────────────────────────────
    cfg.vla_path = cfg.vla_path.rstrip("/")
    chkpt_match = re.search(r"--(\d+)_chkpt", cfg.vla_path)
    if chkpt_match:
        resume = True
        resume_step = int(chkpt_match.group(1))
    elif cfg.vla_path == "./AsyncVLA_release":
        resume = True
        resume_step = 750000
    else:
        resume = False
        resume_step = None
    print(f"resume={resume}, resume_step={resume_step}")

    # ── shead ───────────────────────────────────────────────────────────
    shead = Edge_adapter(
        obs_encoding_size=config["obs_encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    if resume and os.path.exists(os.path.join(cfg.vla_path, f"shead--{resume_step}_checkpoint.pt")):
        ckpt = torch.load(os.path.join(cfg.vla_path, f"shead--{resume_step}_checkpoint.pt"),
                          map_location="cpu")
        if any(k.startswith("module.") for k in ckpt.keys()):
            ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        shead.load_state_dict(ckpt, strict=False)
        print("shead loaded.")
    shead.to(torch.bfloat16).to(device_id)
    shead = wrap_ddp(shead, device_id, find_unused=True)
    shead.eval()

    # ── VLA ─────────────────────────────────────────────────────────────
    Load_hf = model_is_on_hf_hub(cfg.vla_path)
    if not Load_hf:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    if distributed_state.is_main_process:
        update_auto_map(cfg.vla_path)
        check_model_logic_mismatch(cfg.vla_path)
    dist.barrier()

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device_id)

    if cfg.use_lora:
        target_modules = [name for name, m in vla.named_modules() if isinstance(m, nn.Linear)]
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)

    vla = wrap_ddp(vla, device_id, find_unused=True)
    vla.eval()

    # ── pose_projector ──────────────────────────────────────────────────
    pose_projector = ProprioProjector(llm_dim=vla.module.llm_dim, proprio_dim=POSE_DIM)
    if resume:
        sd = load_checkpoint("pose_projector", cfg.vla_path, resume_step)
        pose_projector.load_state_dict(sd)
    pose_projector.to(device_id)
    pose_projector = wrap_ddp(pose_projector, device_id)
    pose_projector.eval()

    # ── action_head ─────────────────────────────────────────────────────
    action_head = L1RegressionActionHead_idcat(
        input_dim=vla.module.llm_dim,
        hidden_dim=vla.module.llm_dim,
        action_dim=ACTION_DIM,
    )
    if resume:
        sd = load_checkpoint("action_head", cfg.vla_path, resume_step)
        action_head.load_state_dict(sd)
    action_head.to(torch.bfloat16).to(device_id)
    action_head = wrap_ddp(action_head, device_id)
    action_head.eval()

    # ── action_proj ─────────────────────────────────────────────────────
    action_proj = Proj_Actiontokens(
        input_dim=vla.module.llm_dim,
        hidden_dim=vla.module.llm_dim,
        action_dim=1024,
    )
    if resume:
        sd = load_checkpoint("action_proj", cfg.vla_path, resume_step)
        action_proj.load_state_dict(sd)
    action_proj.to(torch.bfloat16).to(device_id)
    action_proj = wrap_ddp(action_proj, device_id)
    action_proj.eval()

    NUM_PATCHES = (vla.module.vision_backbone.get_num_patches()
                   * vla.module.vision_backbone.get_num_images_in_input() + 1)

    # ── dataset ─────────────────────────────────────────────────────────
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    collator = PaddedCollatorForActionPrediction_Nav_MMN(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
        num_img=cfg.num_images_in_input,
    )
    dataset = GotoSim_Dataset(
        root_dir=cfg.data_root_dir,
        image_transform=processor.image_processor.apply_transform,
        action_tokenizer=action_tokenizer,
        prompt_builder_fn=PurePromptBuilder,
        base_tokenizer=processor.tokenizer,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=device_id,
                                 shuffle=True, seed=cfg.inference_seed or 0)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=4, drop_last=True, sampler=sampler)

    # ── inference loop ───────────────────────────────────────────────────
    sample_count = 0
    for batch in loader:
        if sample_count >= cfg.num_samples:
            break

        batch = merge_batches_padding([batch], processor.tokenizer.pad_token_id,
                                      processor.tokenizer.model_max_length)
        modality_id = batch["goal_mask_select"]

        img_cur = transform(batch["c_image"]).to(device_id).to(torch.bfloat16)
        img_past = transform(batch["p_image"]).to(device_id).to(torch.bfloat16)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    attention_mask_label=batch["attention_mask_label"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    modality_id=modality_id.to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                    output_hidden_states=True,
                    proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
                    proprio_projector=pose_projector,
                    use_film=False,
                )

            gt_token_ids = batch["labels"][:, 1:].to(device_id)
            cur_mask = get_current_action_mask(gt_token_ids)
            nxt_mask = get_next_actions_mask(gt_token_ids)
            last_hidden = output.hidden_states[-1]
            text_hidden = last_hidden[:, NUM_PATCHES:-1]
            bsz = batch["input_ids"].shape[0]
            actions_hidden = (
                text_hidden[cur_mask | nxt_mask]
                .reshape(bsz, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
            )

            # VLM raw
            base_actions = action_head.module.predict_action(
                actions_hidden.detach(),
                modality_id.to(torch.bfloat16).to(device_id)
            )

            # Edge adapter
            projected = action_proj.module.predict_action(
                actions_hidden.detach(),
                modality_id.to(torch.bfloat16).to(device_id)
            )
            pred_dactions_cur = shead(img_cur, img_past, projected)
            pred_dactions_past = shead(img_past, img_past, projected)

        gt_actions = batch["actions"].to(torch.float32)
        pred_cur = delta_to_pose(pred_dactions_cur).cpu().to(torch.float32)
        pred_past = delta_to_pose(pred_dactions_past).cpu().to(torch.float32)
        base_pred = base_actions.cpu().to(torch.float32)

        # visualize each sample in batch
        for i in range(bsz):
            if sample_count >= cfg.num_samples:
                break
            save_path = os.path.join(cfg.output_dir, f"sample_{sample_count:04d}.png")
            visualize_sample(
                batch_past_img=batch["p_image"][i],
                batch_curr_img=batch["c_image"][i],
                batch_goal_PIL=batch["gimg_PIL"][i],
                goal_pos_lan=batch["obj_pose_norm"][i],
                goal_pos=batch["goal_pose"][i],
                traj_gt=gt_actions[i],
                traj_vlm_base=base_pred[i],
                traj_edge_past=pred_past[i],
                traj_edge_curr=pred_cur[i],
                goal_mask_select=batch["goal_mask_select"][i],
                lan_prompts=batch["lan_prompts"][i] if "lan_prompts" in batch else "",
                save_path=save_path,
            )
            sample_count += 1

    print(f"\nDone. Saved {sample_count} visualizations to {cfg.output_dir}")


if __name__ == "__main__":
    main()