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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_tensor

sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/"
])

# prismatic/__init__.py 와 prismatic/vla/__init__.py 는 학습용 heavy import를
# 포함하므로, edge에서 필요한 small_head / constants 만 로드하도록 stub으로 우회
import types as _types
_prismatic_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _pkg, _subpath in [
    ("prismatic",         "prismatic"),
    ("prismatic.vla",     "prismatic/vla"),
    ("prismatic.models",  "prismatic/models"),
]:
    if _pkg not in sys.modules:
        _m = _types.ModuleType(_pkg)
        _m.__path__ = [os.path.join(_prismatic_root, _subpath)]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

from prismatic.models.small_head import Edge_adapter


# ===============================================================
# AMCL / TF helpers
# ===============================================================
def quat_xyzw_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class AMCLPoseProvider:
    """
    Provides robot pose via TF map→base_link (AMCL-corrected).
    Camera images are obtained separately from IsaacSim socket.
    """

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from tf2_ros import Buffer, TransformListener

        self._lock = threading.Lock()
        self._pose: Optional[dict] = None

        provider = self

        class _InnerNode(Node):
            def __init__(inner):
                super().__init__("amcl_pose_provider")
                inner.tf_buffer = Buffer()
                inner.tf_listener = TransformListener(inner.tf_buffer, inner)

                from geometry_msgs.msg import PoseWithCovarianceStamped
                inner._initialpose_pub = inner.create_publisher(
                    PoseWithCovarianceStamped, "/initialpose", 10
                )
                inner.create_timer(0.05, inner._poll_tf)  # 20Hz TF poll

            def _poll_tf(inner):
                import rclpy.time as _rt
                try:
                    tf = inner.tf_buffer.lookup_transform(
                        "map", "base_link", _rt.Time()
                    )
                    tx = tf.transform.translation.x
                    ty = tf.transform.translation.y
                    q = tf.transform.rotation
                    pose = {"x": tx, "y": ty, "yaw": quat_xyzw_to_yaw(q.x, q.y, q.z, q.w)}
                    with provider._lock:
                        provider._pose = pose
                except Exception:
                    pass

        rclpy.init()
        self._node = _InnerNode()
        self._thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True, name="amcl-spin"
        )
        self._thread.start()

    def get_pose(self) -> Optional[dict]:
        with self._lock:
            return dict(self._pose) if self._pose is not None else None

    def publish_initial_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        repeat: int = 3,
        gap_sec: float = 0.8,
        settle_sec: float = 5.0,
    ):
        from geometry_msgs.msg import PoseWithCovarianceStamped
        half = yaw * 0.5
        for i in range(repeat):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.pose.pose.position.x = float(x)
            msg.pose.pose.position.y = float(y)
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation.z = math.sin(half)
            msg.pose.pose.orientation.w = math.cos(half)
            msg.pose.covariance[0]  = 0.25
            msg.pose.covariance[7]  = 0.25
            msg.pose.covariance[35] = 0.0685
            self._node._initialpose_pub.publish(msg)
            print(f"[AMCL] publish_initial_pose ({i+1}/{repeat})  x={x:.2f} y={y:.2f} yaw={yaw:.2f}")
            if i < repeat - 1:
                time.sleep(gap_sec)
        print(f"[AMCL] settling {settle_sec:.1f}s ...")
        time.sleep(settle_sec)

    def close(self):
        import rclpy
        rclpy.shutdown()


# ===============================================================
# Goal definitions
# ===============================================================
goal_poses = {
    "forklift": (-2.71, -2.45143, 2.45),
    "marker1":  (-20.11, 7.0, 1.57),
    "marker2":  (-15.36, 7.0, 1.57),
    "marker3":  (-10.47, 7.0, 1.57),
    "marker4":  (-5.47, 7.0, 1.57),
    "marker5":  (-0.6, 7.0, 1.57),
    "pallet":   (0.54, -13.29, 0.31),
}

goal_image_paths = {
    "forklift": "./inference/finetune_model/goal_img/forklift.png",
    "marker1":  "./inference/finetune_model/goal_img/marker1.png",
    "marker2":  "./inference/finetune_model/goal_img/marker2.png",
    "marker3":  "./inference/finetune_model/goal_img/marker3.png",
    "marker4":  "./inference/finetune_model/goal_img/marker4.png",
    "marker5":  "./inference/finetune_model/goal_img/marker5.png",
    "pallet":   "./inference/finetune_model/goal_img/pallet.png",
}

WAYPOINT_SPACING = 0.25

transform = transforms.Compose([
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


# ===============================================================
# Socket helpers
# ===============================================================
class JsonSocketClient:
    """
    Blocking request/response client.
    Handles partial recv by looping until a newline delimiter is received.
    """

    def __init__(self, host: str, port: int, recv_buf: int = 4 * 1024 * 1024):
        self.host = host
        self.port = port
        self.recv_buf = recv_buf
        self.sock: Optional[socket.socket] = None
        self.buffer = b""
        self._connect()

    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buf)
        self.buffer = b""

    def request(self, payload: dict) -> dict:
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        # Loop until we have a complete newline-delimited response
        while b"\n" not in self.buffer:
            data = self.sock.recv(1 << 20)
            if not data:
                raise RuntimeError("Server disconnected during recv")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


class JsonLineSender:
    """
    Fire-and-forget sender for cmd_vel bridge.
    Sends one JSON line per action. Auto-reconnects on failure.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self._connect()

    def _connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port))
        except OSError as e:
            print(f"[CMD SENDER] connect failed: {e}")
            self.sock = None

    def send(self, payload: dict):
        if self.sock is None:
            self._connect()
        if self.sock is None:
            return
        try:
            self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"[CMD SENDER] send failed, reconnecting: {e}")
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


# ===============================================================
# Model helpers
# ===============================================================
def remove_ddp_in_checkpoint(state_dict: dict) -> dict:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    """
    delta: [N, T, 4]  (dx, dy, cos_dtheta, sin_dtheta)
    return: [N, T, 4] cumulative pose chunk in robot frame
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
    vla_path: str = "/nas/sujinkim/model/goto/sim/20260323_224/AsyncVLA+2_step_trainig__STEP2+more_delay+no_lan/omnivla-original-balance--550000_chkpt-merged/"
    resume_step: Optional[int] = 550000


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
    print("Loading shead checkpoint:", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = remove_ddp_in_checkpoint(state_dict)

    missing, unexpected = shead.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

    shead = shead.to(torch.bfloat16).to(device).eval()
    return shead, device


# ===============================================================
# Async base requester
# ===============================================================
class AsyncBaseRequester:
    """
    Decouples base inference requests from the 8Hz edge loop.

    Design:
    - One background worker thread owns the TCP connection to base server.
    - Worker blocks on recv() while inference is running — that is expected.
    - Edge loop calls enqueue_latest() at 5Hz; only the newest request is kept.
    - Edge loop calls pop_latest_response() each step to drain results.
    - On any socket error the worker auto-reconnects before the next request.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

        self._client: Optional[JsonSocketClient] = None
        self._client_lock = threading.Lock()

        self._req_queue: "queue.Queue[dict]" = queue.Queue(maxsize=2)
        self._latest_response: Optional[dict] = None
        self._resp_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="base-worker")
        self._worker.start()

    # ----------------------------------------------------------
    def _get_or_connect(self) -> Optional[JsonSocketClient]:
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                self._client = JsonSocketClient(self.host, self.port)
                print(f"[ASYNC BASE] connected to {self.host}:{self.port}")
            except OSError as e:
                print(f"[ASYNC BASE] connect failed: {e}")
                self._client = None
            return self._client

    def _invalidate_connection(self):
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    # ----------------------------------------------------------
    def _run(self):
        while not self._stop_event.is_set():
            try:
                payload = self._req_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            client = self._get_or_connect()
            if client is None:
                # No connection — drop this request and wait before retrying
                self._req_queue.task_done()
                time.sleep(0.5)
                continue

            try:
                t0 = time.perf_counter()
                resp = client.request(payload)
                elapsed = (time.perf_counter() - t0) * 1000.0

                with self._resp_lock:
                    self._latest_response = resp

                print(
                    f"[ASYNC BASE] response received | ok={resp.get('ok')} "
                    f"ts={resp.get('timestamp', 'n/a')} "
                    f"modality={resp.get('modality_id', 'n/a')} "
                    f"rtt={elapsed:.0f}ms"
                )

            except Exception as e:
                print(f"[ASYNC BASE] request error: {e} — reconnecting")
                self._invalidate_connection()
                with self._resp_lock:
                    self._latest_response = {"ok": False, "error": str(e)}
            finally:
                self._req_queue.task_done()

    # ----------------------------------------------------------
    def enqueue_latest(self, payload: dict) -> bool:
        """
        Always keep only the most recent request in the queue.
        If an older request is waiting, discard it and put the new one.
        Returns True if successfully enqueued.
        """
        # Drain any stale pending requests
        drained = 0
        while True:
            try:
                self._req_queue.get_nowait()
                self._req_queue.task_done()
                drained += 1
            except queue.Empty:
                break
        if drained:
            print(f"[ASYNC BASE] dropped {drained} stale request(s) from queue")

        try:
            self._req_queue.put_nowait(payload)
            return True
        except queue.Full:
            # Worker picked up the old item in the tiny race window — not a problem
            return False

    def pop_latest_response(self) -> Optional[dict]:
        """Return and clear the latest base response. Returns None if nothing new."""
        with self._resp_lock:
            resp = self._latest_response
            self._latest_response = None
        return resp
    
    def flush_response(self):
        while True:
            try:
                self._req_queue.get_nowait()
                self._req_queue.task_done()
            except queue.Empty:
                break
        with self._resp_lock:
            self._latest_response = None
        time.sleep(0.2)  # worker가 현재 recv 중인 응답 한 개 소화 대기
        with self._resp_lock:
            self._latest_response = None  # 한 번 더 버림

    def close(self):
        self._stop_event.set()
        self._worker.join(timeout=2.0)
        self._invalidate_connection()


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

        self.obs_buffer: deque = deque(maxlen=obs_buffer_size)  # (timestamp, image_PIL, pose_dict)
        self.cached_projected_actions: Optional[torch.Tensor] = None
        self.cached_embedding_timestamp: Optional[float] = None
        self.cached_modality_id = np.array([5], dtype=np.int64)

    def get_next_episode_index(self) -> int:
        existing = []
        for name in os.listdir(self.base_save_dir):
            full_path = os.path.join(self.base_save_dir, name)
            if os.path.isdir(full_path) and name.isdigit():
                existing.append(int(name))
        return 0 if not existing else max(existing) + 1

    def set_episode_save_dir(self, episode_idx: int):
        self.datastore_path_image = os.path.join(self.base_save_dir, f"{episode_idx:03d}")
        os.makedirs(self.datastore_path_image, exist_ok=True)
        self.count_id = 0
        self.obs_buffer.clear()
        self.cached_projected_actions = None
        self.cached_embedding_timestamp = None
        self.cached_modality_id = np.array([5], dtype=np.int64)
        print(f"[EDGE] episode save dir: {self.datastore_path_image}")

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
        goal_distance = math.sqrt(x_rel ** 2 + y_rel ** 2)

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

    def append_observation(self, pose_dict: dict, image_b64: str, timestamp: float) -> Image.Image:
        img = self.decode_image_b64(image_b64)
        self.obs_buffer.append((float(timestamp), img, dict(pose_dict)))
        return img

    def get_latest_observation(self) -> Optional[tuple]:
        if len(self.obs_buffer) == 0:
            return None
        return self.obs_buffer[-1]

    def find_buffer_item_by_timestamp(self, target_ts: float) -> Optional[tuple]:
        if len(self.obs_buffer) == 0:
            return None
        best_item = None
        best_dt = float("inf")
        for item in self.obs_buffer:
            dt = abs(item[0] - target_ts)
            if dt < best_dt:
                best_dt = dt
                best_item = item
        return best_item

    # ----------------------------------------------------------
    # Edge forward
    # ----------------------------------------------------------
    def run_edge_forward(
        self,
        projected_actions: torch.Tensor,
        delayed_image_PIL: Image.Image,
        current_image_PIL: Image.Image,
    ) -> torch.Tensor:
        p_image = TF.resize(to_tensor(delayed_image_PIL), (96, 96)).unsqueeze(0)
        c_image = TF.resize(to_tensor(current_image_PIL), (96, 96)).unsqueeze(0)

        img_past = transform(p_image).to(self.device).to(torch.bfloat16)
        img_cur  = transform(c_image).to(self.device).to(torch.bfloat16)

        with torch.no_grad():
            predicted_dactions = self.shead(img_cur, img_past, projected_actions)
            predicted_actions = delta_to_pose(predicted_dactions)

        return predicted_actions

    # ----------------------------------------------------------
    # Policy / PD controller
    # ----------------------------------------------------------
    def compute_cmd_vel_from_waypoint(self, dx: float, dy: float, hx: float, hy: float):
        EPS = 1e-8

        POS_DEADBAND  = 0.03
        YAW_DEADBAND  = 0.10
        SLOW_RADIUS   = 0.25

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
                return 0.0, 0.0
            else:
                return 0.0, float(np.clip(KP_ANG * heading_error, -0.25, 0.25))

        slow_scale = min(1.0, dist / SLOW_RADIUS)
        linear_vel_value = np.clip(KP_LIN * dx * slow_scale, -MAXV, MAXV)

        angular_scale = min(1.0, max(0.3, dist / SLOW_RADIUS))
        angular_vel_value = np.clip(KP_ANG * path_angle * angular_scale, -MAXW, MAXW)

        # Joint velocity limit enforcement
        if abs(linear_vel_value) <= MAXV:
            if abs(angular_vel_value) <= MAXW:
                return float(linear_vel_value), float(angular_vel_value)
            else:
                rd = linear_vel_value / (angular_vel_value + EPS)
                return float(MAXW * np.sign(linear_vel_value) * abs(rd)), float(MAXW * np.sign(angular_vel_value))
        else:
            if abs(angular_vel_value) <= 1e-3:
                return float(MAXV * np.sign(linear_vel_value)), 0.0
            else:
                rd = linear_vel_value / angular_vel_value
                if abs(rd) >= MAXV / MAXW:
                    return (
                        float(MAXV * np.sign(linear_vel_value)),
                        float(MAXV * np.sign(angular_vel_value) / abs(rd)),
                    )
                else:
                    return (
                        float(MAXW * np.sign(linear_vel_value) * abs(rd)),
                        float(MAXW * np.sign(angular_vel_value)),
                    )

    def make_base_request_payload(self, pose_dict: dict, image_b64: str, timestamp: float, episode_idx: int) -> dict:
        robot_pose_world = (
            float(pose_dict["x"]),
            float(pose_dict["y"]),
            float(pose_dict["yaw"]),
        )
        goal_pose_loc_norm, _ = self._world_to_relative_pose(robot_pose_world, self.goal_pose)
        return {
            "cmd": "infer_base",
            "episode_idx": episode_idx,
            "timestamp": float(timestamp),
            "image_b64": image_b64,
            "goal_pose_loc_norm": goal_pose_loc_norm.tolist(),
            "goal_name": self.goal,
            "lan_inst_prompt": self.lan_inst_prompt,
        }

    def update_cached_base_result(self, base_resp: Optional[dict], current_episode_idx: int) -> bool:
        if base_resp is None:
            return False
        if not base_resp.get("ok", False):
            print(f"[EDGE] base response error: {base_resp.get('error', 'unknown')}")
            return False
        if base_resp.get("episode_idx") != current_episode_idx:
            print(f"[EDGE] stale base response discarded (ep={base_resp.get('episode_idx')} != {current_episode_idx})")
            return False

        self.cached_projected_actions = (
            torch.tensor(
                base_resp["projected_actions"],
                dtype=torch.float32,
                device=self.device,
            )
            .unsqueeze(0)
            .to(torch.bfloat16)
        )
        self.cached_embedding_timestamp = float(base_resp["timestamp"])
        self.cached_modality_id = np.array([base_resp["modality_id"]], dtype=np.int64)
        return True

    def run_policy_from_latest(self, save: bool = True) -> Tuple[float, float, float, bool, bool]:
        latest_item = self.get_latest_observation()
        if latest_item is None:
            return 0.0, 0.0, 0.0, False, False

        latest_ts, current_image_PIL, current_pose = latest_item
        robot_pose_world = (
            float(current_pose["x"]),
            float(current_pose["y"]),
            float(current_pose["yaw"]),
        )
        goal_pose_loc_norm, goal_distance = self._world_to_relative_pose(
            robot_pose_world, self.goal_pose
        )

        if self.cached_projected_actions is None or self.cached_embedding_timestamp is None:
            return 0.0, 0.0, float(goal_distance), False, False

        delayed_item = self.find_buffer_item_by_timestamp(self.cached_embedding_timestamp)
        if delayed_item is None:
            delayed_ts = latest_ts
            delayed_image_PIL = current_image_PIL
        else:
            delayed_ts, delayed_image_PIL, _ = delayed_item

        predicted_actions_past = self.run_edge_forward(
            projected_actions=self.cached_projected_actions,
            delayed_image_PIL=delayed_image_PIL,   # img_cur 자리에 past
            current_image_PIL=delayed_image_PIL,   # img_past 고정
        )  # shead(past, past, proj)

        predicted_actions_cur = self.run_edge_forward(
            projected_actions=self.cached_projected_actions,
            delayed_image_PIL=current_image_PIL,   # img_cur 자리에 cur
            current_image_PIL=delayed_image_PIL,   # img_past 고정
        )  # shead(cur, past, proj)

        waypoints = predicted_actions_cur.float().cpu().numpy()
        waypoint_select = 4
        chosen_waypoint = waypoints[0][waypoint_select].copy()
        chosen_waypoint[:2] *= self.metric_waypoint_spacing
        dx, dy, hx, hy = chosen_waypoint

        cmd_vel_v, cmd_vel_w = self.compute_cmd_vel_from_waypoint(dx, dy, hx, hy)
        
        if save:
            self.save_robot_behavior(
                current_image_PIL=current_image_PIL,
                delayed_image_PIL=delayed_image_PIL,
                goal_img=self.goal_image_PIL,
                goal_pose=goal_pose_loc_norm,
                predicted_actions=predicted_actions_cur[0].float().cpu().numpy(),
                predicted_actions_past=predicted_actions_past[0].float().cpu().numpy(),
                linear_vel=float(cmd_vel_v),
                angular_vel=float(cmd_vel_w),
                mask_number=self.cached_modality_id,
                current_ts=latest_ts,
                delayed_ts=delayed_ts,
                embedding_ts=self.cached_embedding_timestamp,
            )

        model_predicts_stop = (
            math.hypot(dx, dy) < 0.08   # 총 이동량 거의 없음
            and abs(dx) < 0.05           # 전진 방향 거의 0
        )
        return float(cmd_vel_v), float(cmd_vel_w), float(goal_distance), True, model_predicts_stop

    # ----------------------------------------------------------
    # Visualization
    # ----------------------------------------------------------
    def save_robot_behavior(
        self,
        current_image_PIL,
        delayed_image_PIL,
        goal_img,
        goal_pose,
        predicted_actions,      # [T, 4]  shead(cur, past, proj) result, numpy
        predicted_actions_past, # [T, 4]  shead(past, past, proj) result, numpy
        linear_vel,
        angular_vel,
        mask_number,
        current_ts,
        delayed_ts,
        embedding_ts,
    ):
        fig = plt.figure(figsize=(16, 10), dpi=100)

        # left col: 3 images stacked | right col: trajectory
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.6], wspace=0.03, hspace=0.08)
        ax_past = fig.add_subplot(gs[0, 0])
        ax_cur  = fig.add_subplot(gs[1, 0])
        ax_goal = fig.add_subplot(gs[2, 0])
        ax_traj = fig.add_subplot(gs[:, 1])

        # ── images ───────────────────────────────────────────────────
        ax_past.imshow(np.array(delayed_image_PIL).astype(np.uint8))
        ax_past.set_title(f"Past (sent to VLM)  ts={delayed_ts:.3f}", fontsize=10)
        ax_past.axis("off")

        ax_cur.imshow(np.array(current_image_PIL).astype(np.uint8))
        ax_cur.set_title(f"Current  ts={current_ts:.3f}", fontsize=10)
        ax_cur.axis("off")

        ax_goal.imshow(np.array(goal_img).astype(np.uint8))
        ax_goal.set_title("Goal image", fontsize=10)
        ax_goal.axis("off")

        # ── trajectories ─────────────────────────────────────────────
        # shead(past, past, proj)  →  past-image corrected
        past_x =  predicted_actions_past[:, 0]
        past_y = -predicted_actions_past[:, 1]

        # shead(cur, past, proj)   →  current-image corrected
        cur_x =  predicted_actions[:, 0]
        cur_y = -predicted_actions[:, 1]

        ax_traj.plot(
            np.insert(past_y, 0, 0.0), np.insert(past_x, 0, 0.0),
            color="royalblue", linewidth=2.5, marker="^", markersize=7,
            label="shead(past, past)  ← past-corrected",
        )
        ax_traj.plot(
            np.insert(cur_y, 0, 0.0), np.insert(cur_x, 0, 0.0),
            color="dodgerblue", linewidth=2.5, marker="o", markersize=7,
            label="shead(cur, past)   ← cur-corrected",
        )
        ax_traj.plot(0, 0, marker="P", color="black", markersize=12, label="Robot")

        # goal star
        mask_type = int(mask_number[0])
        if mask_type in (1, 3, 4, 5, 8):
            ax_traj.plot(
                -float(goal_pose[1]), float(goal_pose[0]),
                marker="*", color="gold", markersize=18,
                markeredgecolor="black", markeredgewidth=0.7,
                label="Goal",
            )

        mask_texts = [
            "satellite only", "pose+satellite", "satellite+image", "all",
            "pose only", "pose+image", "image only", "language only", "language+pose",
        ]
        modality_str = mask_texts[mask_type] if mask_type < len(mask_texts) else str(mask_type)

        ax_traj.set_title(
            f"[{modality_str}]  v={linear_vel:.3f}  w={angular_vel:.3f}  emb_ts={embedding_ts:.3f}",
            fontsize=11,
        )
        ax_traj.set_xlabel("← Left / Right →")
        ax_traj.set_ylabel("← Back / Front →")
        ax_traj.set_xlim(-3.0, 3.0)
        ax_traj.set_ylim(-0.1, 10.0)
        ax_traj.axhline(0, color="gray", lw=0.7, ls="--")
        ax_traj.axvline(0, color="gray", lw=0.7, ls="--")
        ax_traj.grid(True, alpha=0.25)
        ax_traj.legend(loc="upper right", fontsize=9)
        ax_traj.set_aspect("equal", adjustable="box")

        save_path = os.path.join(
            self.datastore_path_image,
            f"{self.count_id:05d}_asyncvla.jpg",
        )
        self.count_id += 1
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

# ===============================================================
# Main (AMCL pose + ROS2 camera)
# ===============================================================
def main():
    ISAACSIM_HOST = "127.0.0.1"
    SIM_PORT      = 8765

    BASE_HOST = "192.168.0.154"
    BASE_PORT = 9001
    CMD_HOST  = "127.0.0.1"       # cmd_vel bridge (same machine as edge)
    CMD_PORT  = 8766

    AMCL_INITIALPOSE_REPEAT   = 3    # /initialpose 반복 발행 횟수
    AMCL_INITIALPOSE_GAP_SEC  = 0.8  # 반복 간격
    AMCL_SETTLE_SEC           = 5.0  # AMCL settle 대기 시간

    HOLD_SECONDS_BEFORE_RESET = 3.0
    HOLD_CMD_DT               = 0.1

    STOP_LINEAR       = 0.02
    STOP_ANGULAR      = 0.02
    STOP_COUNT_THRESH = 10000

    EDGE_HZ    = 8.0
    BASE_HZ    = 5.0
    EDGE_PERIOD = 1.0 / EDGE_HZ
    BASE_PERIOD = 1.0 / BASE_HZ

    ARRIVAL_DISTANCE_THRESH = 0.7
    ARRIVAL_STOP_STEPS      = 15
    WARMUP_STEPS            = int(EDGE_HZ * 2)

    amcl         = AMCLPoseProvider()
    sim          = JsonSocketClient(ISAACSIM_HOST, SIM_PORT)
    cmd_sender   = JsonLineSender(CMD_HOST, CMD_PORT)
    base_req     = AsyncBaseRequester(BASE_HOST, BASE_PORT)
    cli          = AsyncVLAEdgeClient(
        control_hz=int(EDGE_HZ),
        base_hz=int(BASE_HZ),
        goal="marker3",
        save_dir="./results",
    )

    episode_idx   = cli.get_next_episode_index()
    cli.set_episode_save_dir(episode_idx)

    global_step   = 0
    episode_step  = 0
    stop_counter  = 0
    last_edge_t   = 0.0
    last_base_send_t = 0.0
    last_obs_warn_t  = 0.0
    warmup_counter   = 0
    arrival_candidate_counter = 0
    arrived = False

    def hold_still(duration_sec: float):
        hold_start = time.time()
        while time.time() - hold_start < duration_sec:
            cmd_sender.send({"linear": 0.0, "angular": 0.0})
            time.sleep(HOLD_CMD_DT)

    def start_new_episode(ep_idx: int):
        nonlocal arrival_candidate_counter, arrived, warmup_counter
        arrival_candidate_counter = 0
        arrived = False
        warmup_counter = 0
        base_req.flush_response()
        cli.set_episode_save_dir(ep_idx)

        # IsaacSim 랜덤 spawn → AMCL 초기 위치 설정
        reset_resp = sim.request({"cmd": "reset"})
        spawn_pose = reset_resp.get("pose", {})
        print(f"[EP {ep_idx:03d}] sim reset → spawn x={spawn_pose.get('x', 0):.2f} "
              f"y={spawn_pose.get('y', 0):.2f} yaw={spawn_pose.get('yaw', 0):.2f}")
        amcl.publish_initial_pose(
            x=spawn_pose["x"],
            y=spawn_pose["y"],
            yaw=spawn_pose["yaw"],
            repeat=AMCL_INITIALPOSE_REPEAT,
            gap_sec=AMCL_INITIALPOSE_GAP_SEC,
            settle_sec=AMCL_SETTLE_SEC,
        )

    try:
        start_new_episode(episode_idx)

        while True:
            now_t = time.time()

            # Keyboard 'r' → manual reset
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                if key == "r":
                    print("[MANUAL RESET]")
                    hold_still(1.0)
                    episode_idx  += 1
                    episode_step  = 0
                    stop_counter  = 0
                    last_edge_t   = 0.0
                    last_base_send_t = 0.0
                    start_new_episode(episode_idx)
                    continue

            # 8Hz rate limiter
            if now_t - last_edge_t < EDGE_PERIOD:
                time.sleep(0.001)
                continue
            last_edge_t = now_t

            # Camera image from IsaacSim
            obs_resp = sim.request({"cmd": "get_obs"})
            if not obs_resp.get("ok", False):
                if now_t - last_obs_warn_t >= 2.0:
                    print(f"[SIM] get_obs failed: {obs_resp.get('error')}")
                    last_obs_warn_t = now_t
                continue

            # Pose from AMCL
            pose = amcl.get_pose()
            if pose is None:
                if now_t - last_obs_warn_t >= 2.0:
                    print("[AMCL] waiting for TF map->base_link ...")
                    last_obs_warn_t = now_t
                continue

            obs_ts = float(obs_resp["timestamp"])
            cli.append_observation(pose, obs_resp["image_b64"], obs_ts)

            base_resp = base_req.pop_latest_response()
            updated = cli.update_cached_base_result(base_resp, episode_idx)
            if updated:
                print(
                    f"[EDGE] base cache updated | emb_ts={cli.cached_embedding_timestamp:.6f} "
                    f"buffer_len={len(cli.obs_buffer)}"
                )

            warmup_counter += 1

            if now_t - last_base_send_t >= BASE_PERIOD and warmup_counter > WARMUP_STEPS:
                payload = cli.make_base_request_payload(
                    pose_dict=pose,
                    image_b64=obs_resp["image_b64"],
                    timestamp=obs_ts,
                    episode_idx=episode_idx,
                )
                if base_req.enqueue_latest(payload):
                    last_base_send_t = now_t

            linear, angular, goal_distance, has_policy, model_predicts_stop = (
                cli.run_policy_from_latest(save=not arrived)
            )

            is_stop_cmd = abs(linear) < STOP_LINEAR and abs(angular) < STOP_ANGULAR
            stop_counter = stop_counter + 1 if is_stop_cmd else 0

            if not arrived:
                if goal_distance < ARRIVAL_DISTANCE_THRESH and model_predicts_stop:
                    arrival_candidate_counter += 1
                else:
                    arrival_candidate_counter = 0

                if arrival_candidate_counter >= ARRIVAL_STOP_STEPS:
                    arrived = True
                    print("\n" + "=" * 60)
                    print(f"  ARRIVED at [{cli.goal}]")
                    print(f"      dist={goal_distance:.3f}m  model_stop=True")
                    print(f"      ep={episode_idx:03d}  step={episode_step:05d}")
                    print("  Press [r] + Enter to reset")
                    print("=" * 60 + "\n")

            if arrived:
                cmd_sender.send({"linear": 0.0, "angular": 0.0})
                episode_step += 1
                global_step  += 1
                continue

            print(
                f"[EP {episode_idx:03d}|EP_STEP {episode_step:05d}|STEP {global_step:07d}] "
                f"obs_ts={obs_ts:.6f} v={linear:.3f} w={angular:.3f} "
                f"dist={goal_distance:.3f} stop={stop_counter} "
                f"policy={has_policy} buf={len(cli.obs_buffer)}"
            )

            cmd_sender.send({"linear": linear, "angular": angular})

            if stop_counter >= STOP_COUNT_THRESH:
                print(
                    f"[DONE][EP {episode_idx:03d}] "
                    f"stop for {STOP_COUNT_THRESH} steps — "
                    f"holding {HOLD_SECONDS_BEFORE_RESET:.1f}s then resetting"
                )
                hold_still(HOLD_SECONDS_BEFORE_RESET)
                episode_idx  += 1
                episode_step  = 0
                stop_counter  = 0
                last_edge_t   = 0.0
                last_base_send_t = 0.0
                start_new_episode(episode_idx)
                continue

            episode_step += 1
            global_step  += 1

    finally:
        try:
            hold_still(0.3)
        except Exception:
            pass
        base_req.close()
        cmd_sender.close()
        sim.close()
        amcl.close()


if __name__ == "__main__":
    main()