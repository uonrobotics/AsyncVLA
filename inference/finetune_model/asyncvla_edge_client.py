# ===============================================================
# [ ASYNCVLA EDGE CLIENT ]
# 역할:
#   - IsaacSim server에서 8Hz로 pose/image/timestamp 관측 받기
#   - observation buffer 유지
#   - remote base server에 5Hz로 delayed observation 전송
#   - base가 돌려준 timestamp로 buffer matching
#   - local edge adapter(shead)로 trajectory refinement
#   - PD controller로 cmd_vel 생성
#   - ROS2 cmd_vel bridge로 action 전송
#
# 논문 스타일 핵심:
#   - fs = 8Hz (edge loop)
#   - fb = 5Hz (base update)
#   - base inference 결과에는 원 timestamp가 붙어 돌아옴
#   - edge는 buffer에서 그 timestamp와 가장 가까운 delayed obs를 찾아 사용
# ===============================================================

import base64
import io
import json
import math
import os
import socket
import sys
import time
import select
import threading
import queue
from collections import deque
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor

from prismatic.models.small_head import Edge_adapter


# ===============================================================
# Goal definitions
# ===============================================================
goal_poses = {
    "forklift": (-2.71, -2.45143, 2.45),
    "marker1": (-20.11, 7.0, 1.57),
    "marker2": (-15.36, 7.0, 1.57),
    "marker3": (-10.47, 7.0, 1.57),
    "marker4": (-5.47, 7.0, 1.57),
    "marker5": (-0.6, 7.0, 1.57),
    "pallet": (0.54, -13.29, 0.31),
}

goal_image_paths = {
    "forklift": "./goal_img/forklift.png",
    "marker1": "./goal_img/marker1.png",
    "marker2": "./goal_img/marker2.png",
    "marker3": "./goal_img/marker3.png",
    "marker4": "./goal_img/marker4.png",
    "marker5": "./goal_img/marker5.png",
    "pallet": "./goal_img/pallet.png",
}

WAYPOINT_SPACING = 0.25

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ===============================================================
# Socket helpers
# ===============================================================
class JsonSocketClient:
    """
    Request/response client for simulator or remote base server.
    Each request expects one JSON line response.
    """

    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port))
        self.buffer = b""

    def request(self, payload: dict) -> dict:
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

        while b"\n" not in self.buffer:
            data = self.sock.recv(10_000_000)
            if not data:
                raise RuntimeError("Server disconnected.")
            self.buffer += data

        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class JsonLineSender:
    """
    Fire-and-forget sender for cmd_vel bridge.
    Sends one JSON line per action.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.sock = socket.create_connection((host, port))

    def send(self, payload: dict):
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# ===============================================================
# Model helpers
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    """
    delta: [N, T, 4]
    return: [N, T, 4] pose chunk
    """
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])

    _, T = dx.shape
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


class EdgeConfig:
    resume: bool = True
    vla_path: str = "./AsyncVLA_release"
    resume_step: Optional[int] = 750000


def define_model(cfg: EdgeConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    with open("./config_nav/dataset_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    shead = Edge_adapter(
        obs_encoding_size=config["obs_encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )

    checkpoint_path = os.path.join(
        cfg.vla_path.rstrip("/"),
        f"shead--{cfg.resume_step}_checkpoint.pt",
    )
    print("Loading checkpoint:", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = remove_ddp_in_checkpoint(state_dict)

    missing, unexpected = shead.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    shead = shead.to(torch.bfloat16).to(device).eval()
    return shead, device


# ===============================================================
# Async base request worker
# ===============================================================
class AsyncBaseRequester:
    """
    base server 요청을 edge loop와 분리하기 위한 worker.
    - 한 번에 하나만 in-flight 허용
    - edge loop는 8Hz 유지
    - base inference는 5Hz target으로 enqueue
    """
    def __init__(self, host: str, port: int):
        self.client = JsonSocketClient(host, port)
        self.req_queue: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self.latest_response = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                payload = self.req_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                resp = self.client.request(payload)
                with self.lock:
                    self.latest_response = resp
            except Exception as e:
                with self.lock:
                    self.latest_response = {"ok": False, "error": str(e)}
            finally:
                self.req_queue.task_done()

    def enqueue_latest(self, payload: dict):
        """
        backlog 쌓지 않고 항상 가장 최신 요청만 남김
        """
        try:
            while True:
                self.req_queue.get_nowait()
                self.req_queue.task_done()
        except queue.Empty:
            pass

        try:
            self.req_queue.put_nowait(payload)
            return True
        except queue.Full:
            return False

    def pop_latest_response(self):
        with self.lock:
            resp = self.latest_response
            self.latest_response = None
        return resp

    def close(self):
        self.stop_event.set()
        self.worker.join(timeout=1.0)
        self.client.close()


# ===============================================================
# AsyncVLA edge client
# ===============================================================
class AsyncVLAEdgeClient:
    def __init__(
        self,
        control_hz: int = 8,
        base_hz: int = 5,
        goal: str = "marker1",
        save_dir: str = "./results",
        obs_buffer_size: int = 64,
    ):
        if goal not in goal_poses:
            raise ValueError(f"Unknown goal: {goal}")

        self.control_hz = control_hz
        self.base_hz = base_hz
        self.goal = goal
        self.goal_pose = goal_poses[goal]
        self.goal_image_PIL = Image.open(goal_image_paths[goal]).convert("RGB")
        self.lan_inst_prompt = "xxxx"
        self.metric_waypoint_spacing = WAYPOINT_SPACING
        self.count_id = 0

        self.base_save_dir = save_dir
        os.makedirs(self.base_save_dir, exist_ok=True)
        self.datastore_path_image = None

        cfg = EdgeConfig()
        self.shead, self.device = define_model(cfg)

        self.obs_buffer = deque(maxlen=obs_buffer_size)  # (timestamp, image_PIL, pose_dict)
        self.cached_projected_actions = None
        self.cached_embedding_timestamp = None
        self.cached_modality_id = np.array([5], dtype=np.int64)

    def get_next_episode_index(self) -> int:
        existing = []
        for name in os.listdir(self.base_save_dir):
            full_path = os.path.join(self.base_save_dir, name)
            if os.path.isdir(full_path) and name.isdigit():
                existing.append(int(name))

        if not existing:
            return 0
        return max(existing) + 1

    def set_episode_save_dir(self, episode_idx: int):
        self.datastore_path_image = os.path.join(self.base_save_dir, f"{episode_idx:03d}")
        os.makedirs(self.datastore_path_image, exist_ok=True)
        self.count_id = 0
        self.obs_buffer.clear()
        self.cached_projected_actions = None
        self.cached_embedding_timestamp = None
        self.cached_modality_id = np.array([5], dtype=np.int64)
        print(f"[SAVE DIR] {self.datastore_path_image}")

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return math.atan2(math.sin(theta), math.cos(theta))

    def _world_to_relative_pose(
        self,
        robot_pose_world: Tuple[float, float, float],
        goal_pose_world: Tuple[float, float, float],
    ) -> Tuple[np.ndarray, float]:
        xr, yr, yaw_r = robot_pose_world
        xg, yg, yaw_g = goal_pose_world

        dx = xg - xr
        dy = yg - yr

        x_rel = math.cos(yaw_r) * dx + math.sin(yaw_r) * dy
        y_rel = -math.sin(yaw_r) * dx + math.cos(yaw_r) * dy
        dyaw = self._wrap_angle(yaw_g - yaw_r)

        goal_distance = math.sqrt(x_rel**2 + y_rel**2)

        x_rel_norm = x_rel / self.metric_waypoint_spacing
        y_rel_norm = y_rel / self.metric_waypoint_spacing

        goal_pose_cos_sin = np.array(
            [x_rel_norm, y_rel_norm, math.cos(dyaw), math.sin(dyaw)],
            dtype=np.float32,
        )
        return goal_pose_cos_sin, goal_distance

    @staticmethod
    def decode_image_b64(image_b64: str) -> Image.Image:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def append_observation(self, pose_dict: dict, image_b64: str, timestamp: float):
        img = self.decode_image_b64(image_b64)
        self.obs_buffer.append((float(timestamp), img, dict(pose_dict)))
        return img

    def get_latest_observation(self):
        if len(self.obs_buffer) == 0:
            return None
        return self.obs_buffer[-1]

    def find_buffer_item_by_timestamp(self, target_ts: float):
        if len(self.obs_buffer) == 0:
            return None

        best_item = None
        best_dt = float("inf")
        for item in self.obs_buffer:
            ts = item[0]
            dt = abs(ts - target_ts)
            if dt < best_dt:
                best_dt = dt
                best_item = item
        return best_item

    # ===========================================================
    # Edge forward
    # ===========================================================
    def run_edge_forward(
        self,
        projected_actions: torch.Tensor,
        delayed_image_PIL: Image.Image,
        current_image_PIL: Image.Image,
    ):
        p_image = TF.resize(to_tensor(delayed_image_PIL), (96, 96)).unsqueeze(0)
        c_image = TF.resize(to_tensor(current_image_PIL), (96, 96)).unsqueeze(0)

        img_past = transform(p_image).to(self.device).to(torch.bfloat16)
        img_cur = transform(c_image).to(self.device).to(torch.bfloat16)

        with torch.no_grad():
            predicted_dactions = self.shead(img_cur, img_past, projected_actions)
            predicted_actions = delta_to_pose(predicted_dactions)

        return predicted_actions

    # ===========================================================
    # Policy
    # ===========================================================
    def compute_cmd_vel_from_waypoint(self, dx: float, dy: float, hx: float, hy: float):
        DT = 1.0 / self.control_hz
        EPS = 1e-8

        POS_DEADBAND = 0.03
        YAW_DEADBAND = 0.10
        SLOW_RADIUS = 0.25

        KP_LIN = 1.2
        KP_ANG = 1.5

        MAXV = 0.8
        MAXW = 0.7

        def clip_angle(theta: float) -> float:
            return math.atan2(math.sin(theta), math.cos(theta))

        heading_error = clip_angle(np.arctan2(hy, hx))
        dist = float(np.hypot(dx, dy))
        path_angle = float(np.arctan2(dy, dx)) if dist > EPS else 0.0

        if dist < POS_DEADBAND:
            if abs(heading_error) < YAW_DEADBAND:
                linear_vel_value_limit = 0.0
                angular_vel_value_limit = 0.0
            else:
                linear_vel_value_limit = 0.0
                angular_vel_value_limit = np.clip(KP_ANG * heading_error, -0.25, 0.25)
        else:
            slow_scale = min(1.0, dist / SLOW_RADIUS)
            linear_vel_value = KP_LIN * dx * slow_scale
            linear_vel_value = np.clip(linear_vel_value, -MAXV, MAXV)

            angular_vel_value = KP_ANG * path_angle
            angular_scale = min(1.0, max(0.3, dist / SLOW_RADIUS))
            angular_vel_value *= angular_scale
            angular_vel_value = np.clip(angular_vel_value, -MAXW, MAXW)

            if np.abs(linear_vel_value) <= MAXV:
                if np.abs(angular_vel_value) <= MAXW:
                    linear_vel_value_limit = linear_vel_value
                    angular_vel_value_limit = angular_vel_value
                else:
                    rd = linear_vel_value / (angular_vel_value + 1e-8)
                    linear_vel_value_limit = MAXW * np.sign(linear_vel_value) * np.abs(rd)
                    angular_vel_value_limit = MAXW * np.sign(angular_vel_value)
            else:
                if np.abs(angular_vel_value) <= 1e-3:
                    linear_vel_value_limit = MAXV * np.sign(linear_vel_value)
                    angular_vel_value_limit = 0.0
                else:
                    rd = linear_vel_value / angular_vel_value
                    if np.abs(rd) >= MAXV / MAXW:
                        linear_vel_value_limit = MAXV * np.sign(linear_vel_value)
                        angular_vel_value_limit = (
                            MAXV * np.sign(angular_vel_value) / np.abs(rd)
                        )
                    else:
                        linear_vel_value_limit = (
                            MAXW * np.sign(linear_vel_value) * np.abs(rd)
                        )
                        angular_vel_value_limit = MAXW * np.sign(angular_vel_value)

        return float(linear_vel_value_limit), float(angular_vel_value_limit)

    def make_base_request_payload(self, pose_dict: dict, image_b64: str, timestamp: float):
        robot_pose_world = (
            float(pose_dict["x"]),
            float(pose_dict["y"]),
            float(pose_dict["yaw"]),
        )
        goal_pose_loc_norm, _ = self._world_to_relative_pose(
            robot_pose_world,
            self.goal_pose,
        )
        return {
            "cmd": "infer_base",
            "timestamp": float(timestamp),
            "image_b64": image_b64,
            "goal_pose_loc_norm": goal_pose_loc_norm.tolist(),
            "goal_name": self.goal,
            "lan_inst_prompt": self.lan_inst_prompt,
        }

    def update_cached_base_result(self, base_resp: dict):
        if base_resp is None:
            return False
        if not base_resp.get("ok", False):
            print("[BASE ERROR]", base_resp)
            return False

        self.cached_projected_actions = torch.tensor(
            base_resp["projected_actions"],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).to(torch.bfloat16)

        self.cached_embedding_timestamp = float(base_resp["timestamp"])
        self.cached_modality_id = np.array([base_resp["modality_id"]], dtype=np.int64)
        return True

    def run_policy_from_latest(self):
        latest_item = self.get_latest_observation()
        if latest_item is None:
            return 0.0, 0.0, 0.0, False

        latest_ts, current_image_PIL, current_pose = latest_item
        robot_pose_world = (
            float(current_pose["x"]),
            float(current_pose["y"]),
            float(current_pose["yaw"]),
        )
        goal_pose_loc_norm, goal_distance = self._world_to_relative_pose(
            robot_pose_world,
            self.goal_pose,
        )

        if self.cached_projected_actions is None or self.cached_embedding_timestamp is None:
            return 0.0, 0.0, float(goal_distance), False

        delayed_item = self.find_buffer_item_by_timestamp(self.cached_embedding_timestamp)
        if delayed_item is None:
            delayed_ts = latest_ts
            delayed_image_PIL = current_image_PIL
        else:
            delayed_ts, delayed_image_PIL, _ = delayed_item

        actions = self.run_edge_forward(
            projected_actions=self.cached_projected_actions,
            delayed_image_PIL=delayed_image_PIL,
            current_image_PIL=current_image_PIL,
        )

        waypoints = actions.float().cpu().numpy()
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= self.metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        cmd_vel_v, cmd_vel_w = self.compute_cmd_vel_from_waypoint(dx, dy, hx, hy)

        self.save_robot_behavior(
            current_image_PIL=current_image_PIL,
            delayed_image_PIL=delayed_image_PIL,
            goal_img=self.goal_image_PIL,
            goal_pose=goal_pose_loc_norm,
            waypoints=waypoints[0],
            linear_vel=float(cmd_vel_v),
            angular_vel=float(cmd_vel_w),
            metric_waypoint_spacing=self.metric_waypoint_spacing,
            mask_number=self.cached_modality_id,
            current_ts=latest_ts,
            delayed_ts=delayed_ts,
            embedding_ts=self.cached_embedding_timestamp,
        )

        return float(cmd_vel_v), float(cmd_vel_w), float(goal_distance), True

    # ===========================================================
    # Visualization
    # ===========================================================
    def save_robot_behavior(
        self,
        current_image_PIL,
        delayed_image_PIL,
        goal_img,
        goal_pose,
        waypoints,
        linear_vel,
        angular_vel,
        metric_waypoint_spacing,
        mask_number,
        current_ts,
        delayed_ts,
        embedding_ts,
    ):
        fig = plt.figure(figsize=(18, 10), dpi=100)
        gs = fig.add_gridspec(2, 2)
        ax_delay = fig.add_subplot(gs[0, 0])
        ax_cur = fig.add_subplot(gs[1, 0])
        ax_graph_pos = fig.add_subplot(gs[:, 1])

        ax_delay.imshow(np.array(delayed_image_PIL).astype(np.uint8))
        ax_cur.imshow(np.array(current_image_PIL).astype(np.uint8))

        x_seq = waypoints[:, 0]
        y_seq_inv = -waypoints[:, 1]
        ax_graph_pos.plot(
            np.insert(y_seq_inv, 0, 0.0),
            np.insert(x_seq, 0, 0.0),
            linewidth=2.0,
            markersize=6,
            marker="o",
        )

        mask_type = int(mask_number[0])
        mask_texts = [
            "satellite only",
            "pose and satellite",
            "satellite and image",
            "all",
            "pose only",
            "pose and image",
            "image only",
            "language only",
            "language and pose",
        ]
        if mask_type < len(mask_texts):
            ax_graph_pos.annotate(
                mask_texts[mask_type],
                xy=(1.0, 0.0),
                xytext=(-20, 20),
                fontsize=12,
                textcoords="offset points",
            )

        ax_delay.set_title(f"Delayed image (match ts={delayed_ts:.3f})")
        ax_cur.set_title(f"Current image (ts={current_ts:.3f})")

        if mask_type in [1, 3, 4, 5, 8]:
            ax_graph_pos.plot(-goal_pose[1], goal_pose[0], marker="*", markersize=12)

        ax_graph_pos.set_xlim(-3.0, 3.0)
        ax_graph_pos.set_ylim(-0.1, 10.0)
        ax_graph_pos.set_title(
            f"Pred trajectory | v={linear_vel:.3f}, w={angular_vel:.3f} | emb_ts={embedding_ts:.3f}"
        )

        save_path = os.path.join(
            self.datastore_path_image,
            f"{self.count_id:05d}_asyncvla.jpg",
        )
        self.count_id += 1
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)


# ===============================================================
# Main
# ===============================================================
def main():
    ISAACSIM_HOST = "192.168.0.180"
    SIM_PORT = 8765
    CMD_PORT = 8766

    BASE_HOST = "192.168.0.154"   # remote base server IP
    BASE_PORT = 9001

    HOLD_SECONDS_BEFORE_RESET = 3.0
    HOLD_CMD_DT = 0.1

    STOP_LINEAR = 0.02
    STOP_ANGULAR = 0.02
    STOP_COUNT_THRESH = 15

    EDGE_HZ = 8.0
    BASE_HZ = 5.0
    EDGE_PERIOD = 1.0 / EDGE_HZ
    BASE_PERIOD = 1.0 / BASE_HZ

    sim = JsonSocketClient(ISAACSIM_HOST, SIM_PORT)
    cmd_sender = JsonLineSender(ISAACSIM_HOST, CMD_PORT)
    base_req = AsyncBaseRequester(BASE_HOST, BASE_PORT)
    cli = AsyncVLAEdgeClient(control_hz=int(EDGE_HZ), base_hz=int(BASE_HZ), goal="marker3", save_dir="./results")

    episode_idx = cli.get_next_episode_index()
    global_step = 0
    episode_step = 0
    stop_counter = 0

    last_edge_t = 0.0
    last_base_send_t = 0.0

    def hold_still(duration_sec: float):
        hold_start = time.time()
        while time.time() - hold_start < duration_sec:
            cmd_sender.send({"linear": 0.0, "angular": 0.0})
            time.sleep(HOLD_CMD_DT)

    def start_new_episode(ep_idx: int):
        cli.set_episode_save_dir(ep_idx)
        reset_resp = sim.request({"cmd": "reset"})
        print(f"[SIM RESET][EP {ep_idx:03d}] {reset_resp}")
        return reset_resp

    try:
        print("[SIM PING]", sim.request({"cmd": "ping"}))
        print("[BASE PING]", JsonSocketClient(BASE_HOST, BASE_PORT).request({"cmd": "ping"}))

        start_new_episode(episode_idx)

        while True:
            now_t = time.time()

            # keyboard reset
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                if key == "r":
                    print("[MANUAL RESET]")

                    hold_still(1.0)

                    episode_idx += 1
                    episode_step = 0
                    stop_counter = 0
                    last_edge_t = 0.0
                    last_base_send_t = 0.0

                    start_new_episode(episode_idx)
                    continue

            # 8Hz edge loop
            if now_t - last_edge_t < EDGE_PERIOD:
                time.sleep(0.001)
                continue

            last_edge_t = now_t

            obs = sim.request({"cmd": "get_obs"})
            if not obs.get("ok", False):
                raise RuntimeError(obs)

            obs_ts = float(obs["timestamp"])
            current_img = cli.append_observation(obs["pose"], obs["image_b64"], obs_ts)

            # async base response polling
            base_resp = base_req.pop_latest_response()
            updated = cli.update_cached_base_result(base_resp)
            if updated:
                print(
                    f"[BASE UPDATE] emb_ts={cli.cached_embedding_timestamp:.6f} "
                    f"buffer_len={len(cli.obs_buffer)}"
                )

            # 5Hz base request trigger
            if now_t - last_base_send_t >= BASE_PERIOD:
                payload = cli.make_base_request_payload(
                    pose_dict=obs["pose"],
                    image_b64=obs["image_b64"],
                    timestamp=obs_ts,
                )
                base_req.enqueue_latest(payload)
                last_base_send_t = now_t

            linear, angular, goal_distance, has_policy = cli.run_policy_from_latest()

            is_stop_cmd = (
                abs(linear) < STOP_LINEAR and abs(angular) < STOP_ANGULAR
            )
            if is_stop_cmd:
                stop_counter += 1
            else:
                stop_counter = 0

            print(
                f"[EP {episode_idx:03d} | EP_STEP {episode_step:05d} | STEP {global_step:07d}] "
                f"obs_ts={obs_ts:.6f} v={linear:.3f}, w={angular:.3f}, "
                f"goal_dist={goal_distance:.3f}, stop_count={stop_counter}, "
                f"has_policy={has_policy}, buffer_len={len(cli.obs_buffer)}"
            )

            cmd_sender.send({"linear": linear, "angular": angular})

            if stop_counter >= STOP_COUNT_THRESH:
                print(
                    f"[DONE][EP {episode_idx:03d}] "
                    f"STOP command detected for {STOP_COUNT_THRESH} consecutive steps. "
                    f"Holding still for {HOLD_SECONDS_BEFORE_RESET:.1f}s before reset..."
                )

                hold_still(HOLD_SECONDS_BEFORE_RESET)

                episode_idx += 1
                episode_step = 0
                stop_counter = 0
                last_edge_t = 0.0
                last_base_send_t = 0.0

                start_new_episode(episode_idx)
                continue

            episode_step += 1
            global_step += 1

    finally:
        try:
            hold_still(0.3)
        except Exception:
            pass

        base_req.close()
        cmd_sender.close()
        sim.close()


if __name__ == "__main__":
    main()