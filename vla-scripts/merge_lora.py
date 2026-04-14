#!/usr/bin/env python3
"""
Offline LoRA merge utility for AsyncVLA / OpenVLA checkpoints.

Supports batch processing with two input modes (can be mixed freely):
  - Parent directory: automatically finds all *_chkpt/ subdirectories and merges each one.
  - Direct checkpoint directory: merges that single checkpoint immediately.

Results are saved to <checkpoint_dir>-merged/.

Typical usage (parent dirs only):
python merge_lora.py \
  --base_model_path /home/sujin/workspace/physical-ai/AsyncVLA/AsyncVLA_release \
  --checkpoint_parent_dirs \
    /nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA \
    /nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA+handle_nan

Typical usage (direct checkpoint dirs only):
python merge_lora.py \
  --base_model_path /home/sujin/workspace/physical-ai/AsyncVLA/AsyncVLA_release \
  --checkpoint_dirs \
    /nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA+2_step_trainig__STEP1/omnivla-original-balance--450000_chkpt/

Typical usage (mixed):
python merge_lora.py \
  --base_model_path /home/sujin/workspace/physical-ai/AsyncVLA/AsyncVLA_release \
  --checkpoint_parent_dirs \
    /nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA+handle_nan \
  --checkpoint_dirs \
    /nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA/step_1000_chkpt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openvla_auto_classes() -> None:
    """Register custom OpenVLA classes for local checkpoints."""
    try:
        AutoConfig.register("openvla", OpenVLAConfig)
    except ValueError:
        pass
    try:
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    except ValueError:
        pass
    try:
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    except ValueError:
        pass
    try:
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)
    except ValueError:
        pass


def copy_non_lora_artifacts(checkpoint_dir: Path, output_dir: Path) -> None:
    """Copy auxiliary AsyncVLA module checkpoints into the merged folder."""
    patterns = [
        "pose_projector--*_checkpoint.pt",
        "proprio_projector--*_checkpoint.pt",
        "action_head--*_checkpoint.pt",
        "action_proj--*_checkpoint.pt",
        "shead--*_checkpoint.pt",
    ]

    copied = []
    for pattern in patterns:
        for src in checkpoint_dir.glob(pattern):
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            copied.append(src.name)

    if copied:
        print("[INFO] Copied auxiliary checkpoints:")
        for name in sorted(copied):
            print(f"  - {name}")
    else:
        print("[WARN] No auxiliary AsyncVLA checkpoints found to copy.")


def save_merge_metadata(output_dir: Path, base_model_path: str, checkpoint_dir: Path) -> None:
    metadata = {
        "base_model_path": base_model_path,
        "checkpoint_dir": str(checkpoint_dir),
        "merged_dtype": "bfloat16",
    }
    with open(output_dir / "merge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def is_checkpoint_dir(path: Path, adapter_subdir: str) -> bool:
    """Return True if path is a direct checkpoint dir (contains the adapter subfolder)."""
    return path.is_dir() and (path / adapter_subdir).exists()


def find_checkpoint_dirs(parent_dir: Path, adapter_subdir: str) -> list[Path]:
    """
    Find all subdirectories inside parent_dir that contain a LoRA adapter folder.
    Matches the pattern *_chkpt/ with a valid adapter_subdir inside.
    """
    candidates = sorted(parent_dir.glob("*_chkpt"))
    valid = [p for p in candidates if p.is_dir() and (p / adapter_subdir).exists()]
    return valid


def collect_checkpoints(
    checkpoint_parent_dirs: list[str],
    checkpoint_dirs: list[str],
    adapter_subdir: str,
) -> list[tuple[Path, Path]]:
    """
    Resolve all checkpoint dirs from both input modes.
    Returns a deduplicated list of (checkpoint_dir, output_dir) pairs.
    """
    seen: set[Path] = set()
    all_checkpoints: list[tuple[Path, Path]] = []

    def add(chkpt_dir: Path) -> None:
        chkpt_dir = chkpt_dir.resolve()
        if chkpt_dir in seen:
            print(f"[WARN] Duplicate checkpoint, skipping: {chkpt_dir}")
            return
        seen.add(chkpt_dir)
        output_dir = chkpt_dir.parent / (chkpt_dir.name + "-merged")
        all_checkpoints.append((chkpt_dir, output_dir))

    # ── Mode 1: parent directories ──────────────────────────────────────────
    for parent_str in checkpoint_parent_dirs:
        parent_dir = Path(parent_str).expanduser().resolve()
        if not parent_dir.exists():
            print(f"[WARN] Parent directory does not exist, skipping: {parent_dir}")
            continue

        chkpt_dirs = find_checkpoint_dirs(parent_dir, adapter_subdir)
        if not chkpt_dirs:
            print(f"[WARN] No valid *_chkpt/ directories found in: {parent_dir}")
            continue

        for chkpt_dir in chkpt_dirs:
            add(chkpt_dir)

    # ── Mode 2: direct checkpoint directories ───────────────────────────────
    for dir_str in checkpoint_dirs:
        chkpt_dir = Path(dir_str).expanduser().resolve()
        if not chkpt_dir.exists():
            print(f"[WARN] Checkpoint directory does not exist, skipping: {chkpt_dir}")
            continue
        if not is_checkpoint_dir(chkpt_dir, adapter_subdir):
            print(
                f"[WARN] '{chkpt_dir}' does not contain adapter subfolder "
                f"'{adapter_subdir}', skipping."
            )
            continue
        add(chkpt_dir)

    return all_checkpoints


def merge_single_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    base_model_path: str,
    adapter_subdir: str,
    save_tokenizer_processor_from: str,
) -> None:
    adapter_dir = checkpoint_dir / adapter_subdir

    print(f"\n{'='*70}")
    print(f"[INFO] Processing checkpoint: {checkpoint_dir.name}")
    print(f"[INFO] Output dir:            {output_dir}")
    print(f"{'='*70}")

    if output_dir.exists():
        print(f"[SKIP] Output already exists, skipping: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading base model from: {base_model_path}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"[INFO] Loading LoRA adapter from: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("[INFO] Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"[INFO] Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)

    processor_source = checkpoint_dir if save_tokenizer_processor_from == "checkpoint" else base_model_path
    print(f"[INFO] Saving processor/tokenizer from: {processor_source}")
    processor = AutoProcessor.from_pretrained(str(processor_source), trust_remote_code=True)
    processor.save_pretrained(output_dir)

    copy_non_lora_artifacts(checkpoint_dir, output_dir)
    save_merge_metadata(output_dir, base_model_path, checkpoint_dir)

    # Free GPU memory before next checkpoint
    del merged_model, peft_model, base_model
    torch.cuda.empty_cache()

    print(f"[DONE] Merged checkpoint written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch offline merge LoRA weights for AsyncVLA")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Base model path or HF repo (e.g. openvla/openvla-7b).",
    )
    parser.add_argument(
        "--checkpoint_parent_dirs",
        type=str,
        nargs="+",
        default=[],
        help=(
            "One or more parent directories containing *_chkpt/ subdirectories. "
            "Each matching checkpoint will be merged and saved as <name>-merged/."
        ),
    )
    parser.add_argument(
        "--checkpoint_dirs",
        type=str,
        nargs="+",
        default=[],
        help=(
            "One or more direct checkpoint directories (each must contain the "
            "adapter subfolder). Merged output is saved as <name>-merged/ "
            "alongside the checkpoint."
        ),
    )
    parser.add_argument(
        "--adapter_subdir",
        type=str,
        default="lora_adapter",
        help="Relative adapter folder name inside each checkpoint dir. Default: lora_adapter",
    )
    parser.add_argument(
        "--save_tokenizer_processor_from",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "base"],
        help="Whether to save processor/tokenizer from checkpoint_dir or base_model_path.",
    )
    args = parser.parse_args()

    if not args.checkpoint_parent_dirs and not args.checkpoint_dirs:
        parser.error("Provide at least one of --checkpoint_parent_dirs or --checkpoint_dirs.")

    register_openvla_auto_classes()

    all_checkpoints = collect_checkpoints(
        checkpoint_parent_dirs=args.checkpoint_parent_dirs,
        checkpoint_dirs=args.checkpoint_dirs,
        adapter_subdir=args.adapter_subdir,
    )

    if not all_checkpoints:
        print("[ERROR] No checkpoints found to process. Exiting.")
        return

    print(f"\n[INFO] Found {len(all_checkpoints)} checkpoint(s) to merge:")
    for chkpt_dir, output_dir in all_checkpoints:
        status = "EXISTS (will skip)" if output_dir.exists() else "pending"
        print(f"  {chkpt_dir.name}  ->  {output_dir.name}  [{status}]")

    for i, (chkpt_dir, output_dir) in enumerate(all_checkpoints, 1):
        print(f"\n[{i}/{len(all_checkpoints)}] Starting merge...")
        try:
            merge_single_checkpoint(
                checkpoint_dir=chkpt_dir,
                output_dir=output_dir,
                base_model_path=args.base_model_path,
                adapter_subdir=args.adapter_subdir,
                save_tokenizer_processor_from=args.save_tokenizer_processor_from,
            )
        except Exception as e:
            print(f"[ERROR] Failed to merge {chkpt_dir.name}: {e}")
            # Clean up incomplete output dir to allow retry
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"[INFO] Removed incomplete output dir: {output_dir}")

    print("\n[ALL DONE] Batch merge completed.")


if __name__ == "__main__":
    main()