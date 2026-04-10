# ===============================================================
# [ ASYNCVLA BASE SERVER ]
# 역할:
#   - edge client에서 delayed observation(image,timestamp) 수신
#   - AsyncVLA base(VLA + pose_projector + action_proj) 추론 수행
#   - projected_actions + modality_id + 원 timestamp 반환
#
# 실행 위치:
#   - remote machine (base VLA가 올라가는 머신)
# ===============================================================

import base64
import io
import json
import os
import sys
import select
import socket
import time
import random
from typing import Optional, Type

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoImageProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.vla.constants import (
    ACTION_DIM,
    NUM_ACTIONS_CHUNK,
    POSE_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)

sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/"
])

from prismatic.models.small_head import Proj_Actiontokens


goal_image_paths = {
    "forklift": "./inference/finetune_model/goal_img/forklift.png",
    "marker1": "./inference/finetune_model/goal_img/marker1.png",
    "marker2": "./inference/finetune_model/goal_img/marker2.png",
    "marker3": "./inference/finetune_model/goal_img/marker3.png",
    "marker4": "./inference/finetune_model/goal_img/marker4.png",
    "marker5": "./inference/finetune_model/goal_img/marker5.png",
    "pallet": "./inference/finetune_model/goal_img/pallet.png",
}

pose_goal = True
satellite = False
image_goal = True
lan_prompt = False


class JsonSocketServer:
    def __init__(self, host="0.0.0.0", port=9001):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.setblocking(False)
        self.client = None
        self.buffer = b""

    def _drop_client(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
        self.buffer = b""

    def poll_accept(self):
        if self.client is not None:
            return
        readable, _, _ = select.select([self.server], [], [], 0.0)
        if readable:
            conn, addr = self.server.accept()
            conn.setblocking(False)
            self.client = conn
            self.buffer = b""
            print(f"[BASE IPC] connected from {addr}")

    def recv_message(self):
        self.poll_accept()
        if self.client is None:
            return None

        try:
            readable, _, exceptional = select.select([self.client], [], [self.client], 0.0)
        except (OSError, ValueError):
            self._drop_client()
            return None

        if exceptional:
            self._drop_client()
            return None

        if not readable:
            return None

        try:
            data = self.client.recv(10_000_000)
        except (BlockingIOError, InterruptedError):
            return None
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._drop_client()
            return None

        if not data:
            self._drop_client()
            return None

        self.buffer += data
        if b"\n" not in self.buffer:
            return None

        line, self.buffer = self.buffer.split(b"\n", 1)
        if not line.strip():
            return None

        return json.loads(line.decode("utf-8"))

    def send_message(self, payload: dict):
        if self.client is None:
            return False
        try:
            self.client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            return True
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._drop_client()
            return False

    def close(self):
        self._drop_client()
        try:
            self.server.close()
        except Exception:
            pass


def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def load_checkpoint(module_name: str, path: str, step: int, device: str = "cpu") -> dict:
    if (
        not os.path.exists(os.path.join(path, f"{module_name}--{step}_checkpoint.pt"))
        and module_name == "pose_projector"
    ):
        module_name = "proprio_projector"

    checkpoint_path = os.path.join(path, f"{module_name}--{step}_checkpoint.pt")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(f"# trainable params in {name}: {num_params}")


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: "InferenceConfig",
    device: torch.device,
    module_args: dict,
    to_bf16: bool = False,
):
    module = module_class(**module_args)
    count_parameters(module, module_name)

    if cfg.resume:
        state_dict = load_checkpoint(
            module_name,
            cfg.vla_path,
            cfg.resume_step,
            device=str(device),
        )
        module.load_state_dict(state_dict)

    if to_bf16:
        module = module.to(torch.bfloat16)

    return module.to(device)


class InferenceConfig:
    resume: bool = True
    vla_path: str = "/nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA/AsyncVLA_release--790000_chkpt-merged/"
    resume_step: Optional[int] = 790000
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0


def define_model(cfg: InferenceConfig):
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Loading AsyncVLA base model `{cfg.vla_path}`")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    print(
        "Detected constants:\n"
        f"\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n"
        f"\tACTION_DIM: {ACTION_DIM}\n"
        f"\tPOSE_DIM: {POSE_DIM}\n"
        f"\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.to(dtype=torch.bfloat16, device=device)

    pose_projector = init_module(
        ProprioProjector,
        "pose_projector",
        cfg,
        device,
        {"llm_dim": vla.llm_dim, "proprio_dim": POSE_DIM},
    )

    action_proj = init_module(
        Proj_Actiontokens,
        "action_proj",
        cfg,
        device,
        {"input_dim": vla.llm_dim, "hidden_dim": vla.llm_dim, "action_dim": 1024},
        to_bf16=True,
    )

    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )
    num_patches += 1  # for goal pose

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    return (
        vla.eval(),
        pose_projector.eval(),
        action_proj.eval(),
        device,
        num_patches,
        action_tokenizer,
        processor,
    )


class AsyncVLABaseServer:
    def __init__(self):
        cfg = InferenceConfig()
        (
            self.vla,
            self.pose_projector,
            self.action_proj,
            self.device,
            self.num_patches,
            self.action_tokenizer,
            self.processor,
        ) = define_model(cfg)

    @staticmethod
    def decode_image_b64(image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def collator_custom(self, instances, model_max_length, pad_token_id):
        IGNORE_INDEX = -100

        input_ids = pad_sequence(
            [inst["input_ids"] for inst in instances],
            batch_first=True,
            padding_value=pad_token_id,
        )
        labels = pad_sequence(
            [inst["labels"] for inst in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        input_ids = input_ids[:, :model_max_length]
        labels = labels[:, :model_max_length]
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = [inst["pixel_values_current"] for inst in instances]
        pixel_values_goal = [inst["pixel_values_goal"] for inst in instances]
        pixel_values = torch.cat(
            (torch.stack(pixel_values), torch.stack(pixel_values_goal)),
            dim=1,
        )

        actions = torch.stack(
            [torch.from_numpy(inst["actions"].copy()) for inst in instances]
        )
        goal_pose = torch.stack(
            [torch.from_numpy(inst["goal_pose"].copy()) for inst in instances]
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "actions": actions,
            "goal_pose": goal_pose,
        }

    def transform_datatype(
        self,
        inst_obj,
        actions,
        goal_pose_cos_sin,
        current_image_PIL,
        goal_image_PIL,
        prompt_builder,
        action_tokenizer,
        base_tokenizer,
        image_transform,
        predict_stop_token=True,
    ):
        IGNORE_INDEX = -100

        current_action = actions[0]
        future_actions = actions[1:]
        future_actions_string = "".join(action_tokenizer(future_actions))
        current_action_string = action_tokenizer(current_action)
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)

        if inst_obj == "xxxx":
            conversation = [
                {"from": "human", "value": "No language instruction"},
                {"from": "gpt", "value": action_chunk_string},
            ]
        else:
            conversation = [
                {
                    "from": "human",
                    "value": f"What action should the robot take to {inst_obj}?",
                },
                {"from": "gpt", "value": action_chunk_string},
            ]

        pb = prompt_builder("openvla")
        for turn in conversation:
            pb.add_turn(turn["from"], turn["value"])

        input_ids = torch.tensor(
            base_tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
        )
        labels = input_ids.clone()
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pixel_values_current = image_transform(current_image_PIL)
        pixel_values_goal = image_transform(goal_image_PIL)

        return {
            "pixel_values_current": pixel_values_current,
            "pixel_values_goal": pixel_values_goal,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": "lelan",
            "actions": actions.astype(np.float32),
            "goal_pose": goal_pose_cos_sin.astype(np.float32),
        }

    def build_batch(
        self,
        current_image_PIL: Image.Image,
        goal_image_PIL: Image.Image,
        goal_pose_loc_norm: np.ndarray,
        lan_inst_prompt: str,
    ):
        actions = np.random.rand(8, 4).astype(np.float32)

        batch_data = self.transform_datatype(
            lan_inst_prompt,
            actions,
            goal_pose_loc_norm,
            current_image_PIL,
            goal_image_PIL,
            prompt_builder=PurePromptBuilder,
            action_tokenizer=self.action_tokenizer,
            base_tokenizer=self.processor.tokenizer,
            image_transform=self.processor.image_processor.apply_transform,
        )

        return self.collator_custom(
            [batch_data],
            self.processor.tokenizer.model_max_length,
            self.processor.tokenizer.pad_token_id,
        )

    @staticmethod
    def get_modality_id():
        if satellite and not lan_prompt and not pose_goal and not image_goal:
            return torch.as_tensor([0], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and not image_goal:
            return torch.as_tensor([1], dtype=torch.float32)
        elif satellite and not lan_prompt and not pose_goal and image_goal:
            return torch.as_tensor([2], dtype=torch.float32)
        elif satellite and not lan_prompt and pose_goal and image_goal:
            return torch.as_tensor([3], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and not image_goal:
            return torch.as_tensor([4], dtype=torch.float32)
        elif not satellite and not lan_prompt and pose_goal and image_goal:
            return torch.as_tensor([5], dtype=torch.float32)
        elif not satellite and not lan_prompt and not pose_goal and image_goal:
            return torch.as_tensor([6], dtype=torch.float32)
        elif not satellite and lan_prompt and not pose_goal and not image_goal:
            return torch.as_tensor([7], dtype=torch.float32)
        elif not satellite and lan_prompt and pose_goal and not image_goal:
            return torch.as_tensor([8], dtype=torch.float32)
        else:
            raise RuntimeError("Unsupported modality combination")

    def infer_base(
        self,
        current_image_PIL: Image.Image,
        goal_image_PIL: Image.Image,
        goal_pose_loc_norm: np.ndarray,
        lan_inst_prompt: str,
    ):
        batch = self.build_batch(
            current_image_PIL=current_image_PIL,
            goal_image_PIL=goal_image_PIL,
            goal_pose_loc_norm=goal_pose_loc_norm,
            lan_inst_prompt=lan_inst_prompt,
        )

        modality_id = self.get_modality_id()
        autocast_enabled = self.device.type == "cuda"

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device),
                modality_id=modality_id.to(torch.bfloat16).to(self.device),
                labels=batch["labels"].to(self.device),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(torch.bfloat16).to(self.device),
                proprio_projector=self.pose_projector,
                use_film=False,
            )

        ground_truth_token_ids = batch["labels"][:, 1:].to(self.device)
        current_action_mask = get_current_action_mask(ground_truth_token_ids)
        next_actions_mask = get_next_actions_mask(ground_truth_token_ids)

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, self.num_patches : -1]
        batch_size = batch["input_ids"].shape[0]

        actions_hidden_states = (
            text_hidden_states[current_action_mask | next_actions_mask]
            .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        with torch.no_grad():
            projected_actions = self.action_proj.predict_action(
                actions_hidden_states,
                modality_id.to(torch.bfloat16).to(self.device),
            )

        return {
            "projected_actions": projected_actions.squeeze(0).float().cpu().tolist(),
            "modality_id": int(modality_id.item()),
        }


def main():
    HOST = "0.0.0.0"
    PORT = 9001

    server = JsonSocketServer(host=HOST, port=PORT)
    base = AsyncVLABaseServer()

    print(f"[BASE SERVER] listening on {HOST}:{PORT}")

    try:
        while True:
            msg = server.recv_message()
            if msg is None:
                continue

            cmd = msg.get("cmd")

            if cmd == "ping":
                server.send_message({"ok": True, "msg": "pong"})

            elif cmd == "infer_base":
                t_recv = time.time()
                
                current_image_PIL = base.decode_image_b64(msg["image_b64"])
                goal_pose_loc_norm = np.asarray(msg["goal_pose_loc_norm"], dtype=np.float32)

                goal_name = msg.get("goal_name", "marker3")
                if goal_name not in goal_image_paths:
                    raise RuntimeError(f"Unknown goal_name: {goal_name}")

                goal_image_PIL = Image.open(goal_image_paths[goal_name]).convert("RGB")
                lan_inst_prompt = msg.get("lan_inst_prompt", "xxxx")

                out = base.infer_base(
                    current_image_PIL=current_image_PIL,
                    goal_image_PIL=goal_image_PIL,
                    goal_pose_loc_norm=goal_pose_loc_norm,
                    lan_inst_prompt=lan_inst_prompt,
                )
                
                # inference 후 랜덤 delay
                elapsed = time.time() - t_recv
                target = random.uniform(0.10, 0.60) # network latency 재현 
                remaining = target - elapsed
                if remaining > 0:
                    time.sleep(remaining)

                server.send_message({
                    "ok": True,
                    "timestamp": float(msg["timestamp"]),
                    **out,
                })

            else:
                server.send_message({"ok": False, "error": f"unknown cmd: {cmd}"})

    finally:
        server.close()


if __name__ == "__main__":
    main()