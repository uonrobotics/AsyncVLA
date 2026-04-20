# ===============================================================
# [ SIMULATOR SERVER ]
# 역할:
#   - stage/env/robot/camera 초기화
#   - 랜덤 spawn reset
#   - 현재 pose 반환
#   - 현재 RGB 반환
#   - observation timestamp 반환 (AsyncVLA paper-style matching용)
# ===============================================================

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "fast_shutdown": True,
    }
)

import base64
import io
import json
import math
import select
import socket
import time
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import omni.timeline
import omni.usd
import yaml

from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import enable_extension

from collections import defaultdict, deque

# ENV_USD_PATH = "/nas/sujinkim/data/goto/sim/goto_warehouse.usd"
ENV_USD_PATH = "/nas/sujinkim/data/goto/sim/goto_warehouse_extra_obstacles.usd"
ROBOT_REL_PATH = "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"

ENV_PRIM_PATH = "/World/env"
ROBOT_ROOT_PRIM_PATH = "/World/Nova_Carter_ROS"
ROBOT_BODY_PRIM_PATH = "/World/Nova_Carter_ROS/chassis_link"
SPAWN_AREA_PRIM_PATH = "/World/env/spawn_area"
DYNAMIC_OBSTACLE_PATH = "/World/env/dynamic_obstacle/obs_001"

DEFAULT_Z = 0.0

FRONT_CAM_CFG = {
    "name": "cam_front",
    "camera_prim_path": "/World/replay_camera/front_camera",
    "resolution": (320, 240),
    "fov_deg": 90.0,
    "offset_xyz": [0.20, 0.0, 0.80],
    "rot_xyz_deg": [90.0, -90.0, 0.0],
}

# ===============================================================
# Latency logging helpers
# ===============================================================
SERVER_PERF = defaultdict(lambda: deque(maxlen=200))
SERVER_OBS_SEQ = 0


def now_perf():
    return time.perf_counter()


def push_server_metric(name: str, value_ms: float):
    SERVER_PERF[name].append(float(value_ms))


def server_metric_last(name: str, default=0.0):
    arr = SERVER_PERF.get(name)
    if not arr:
        return default
    return arr[-1]


def server_metric_mean(name: str, default=0.0):
    arr = SERVER_PERF.get(name)
    if not arr:
        return default
    return float(np.mean(arr))


def server_metric_p95(name: str, default=0.0):
    arr = SERVER_PERF.get(name)
    if not arr:
        return default
    return float(np.percentile(np.asarray(arr, dtype=np.float64), 95))


def print_server_latency_summary():
    print(
        "[SERVER][LAT SUMMARY] "
        f"sensor_total(mean/p95)={server_metric_mean('srv.obs_total_ms'):.1f}/{server_metric_p95('srv.obs_total_ms'):.1f}ms | "
        f"rgba(mean/p95)={server_metric_mean('srv.rgba_ms'):.1f}/{server_metric_p95('srv.rgba_ms'):.1f}ms | "
        f"enc(mean/p95)={server_metric_mean('srv.encode_ms'):.1f}/{server_metric_p95('srv.encode_ms'):.1f}ms"
    )


def add_physics_scene(stage):
    scene_path = "/physicsScene"
    if stage.GetPrimAtPath(scene_path).IsValid():
        return
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBroadphaseTypeAttr("MBP")


def add_dome_light(stage):
    dome_path = "/World/DomeLight"
    if stage.GetPrimAtPath(dome_path).IsValid():
        return
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(1000)


def get_valid_prim(stage, prim_path: str, name: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"{name} prim not found: {prim_path}")
    return prim


def get_world_xy_yaw(stage, prim_path: str):
    prim = get_valid_prim(stage, prim_path, prim_path)
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    pos = mat.ExtractTranslation()
    world_forward = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    yaw = math.atan2(float(world_forward[1]), float(world_forward[0]))
    return float(pos[0]), float(pos[1]), float(yaw)


def get_world_bbox_xy(stage, prim_path: str):
    prim = get_valid_prim(stage, prim_path, prim_path)
    bbox_cache = UsdGeom.BBoxCache(0, ["default"])
    bound = bbox_cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedBox()
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    return float(min_pt[0]), float(max_pt[0]), float(min_pt[1]), float(max_pt[1])


def quat_wxyz_from_yaw(yaw_rad: float):
    half = yaw_rad * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def sample_xy_in_region(region, margin=0.3):
    xmin, xmax, ymin, ymax = region
    x = np.random.uniform(xmin + margin, xmax - margin)
    y = np.random.uniform(ymin + margin, ymax - margin)
    return float(x), float(y)


class OccupancyDistanceMap:
    def __init__(self, map_png_path, map_yaml_path):
        with open(map_yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.resolution = float(cfg["resolution"])
        self.origin_x = float(cfg["origin"][0])
        self.origin_y = float(cfg["origin"][1])

        img = Image.open(map_png_path).convert("L")
        self.map_img = np.array(img)
        self.occupied = self.map_img < 128
        self.height, self.width = self.occupied.shape

        from scipy.ndimage import distance_transform_edt
        free_mask = ~self.occupied
        dist_pixels = distance_transform_edt(free_mask)
        self.dist_meters = dist_pixels * self.resolution

    def world_to_map_rc(self, x, y):
        mx = int((x - self.origin_x) / self.resolution)
        my = int((y - self.origin_y) / self.resolution)
        row = self.height - 1 - my
        col = mx
        return row, col

    def is_inside(self, x, y):
        row, col = self.world_to_map_rc(x, y)
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, x, y):
        if not self.is_inside(x, y):
            return False
        row, col = self.world_to_map_rc(x, y)
        return not self.occupied[row, col]

    def clearance(self, x, y):
        if not self.is_inside(x, y):
            return 0.0
        row, col = self.world_to_map_rc(x, y)
        return float(self.dist_meters[row, col])


def sample_conditioned_spawn(region, occ_map, min_clearance=1.5, max_trials=50):
    for _ in range(max_trials):
        x, y = sample_xy_in_region(region, margin=0.3)
        if not occ_map.is_inside(x, y):
            continue
        if not occ_map.is_free(x, y):
            continue
        if occ_map.clearance(x, y) < min_clearance:
            continue
        yaw = np.random.uniform(-math.pi, math.pi)
        return x, y, float(yaw)
    raise RuntimeError("Failed to sample valid spawn pose.")


# ===============================================================
# JSON Socket Server
# ===============================================================
class JsonSocketServer:
    """
    Non-blocking TCP server that:
    - accepts one client at a time
    - reads newline-delimited JSON messages from the client
    - handles partial recv correctly by accumulating into self.buffer
    - auto-drops client on any socket error
    """

    def __init__(self, host="127.0.0.1", port=8765):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.setblocking(False)
        self.client: Optional[socket.socket] = None
        self.buffer = b""

    def _drop_client(self, reason: str = ""):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.buffer = b""
            if reason:
                print(f"[SIM IPC] client dropped: {reason}")

    def poll_accept(self):
        if self.client is not None:
            return
        readable, _, _ = select.select([self.server], [], [], 0.0)
        if readable:
            try:
                conn, addr = self.server.accept()
                conn.setblocking(False)
                # Large receive buffer for image payloads
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                self.client = conn
                self.buffer = b""
                print(f"[SIM IPC] client connected from {addr}")
            except OSError as e:
                print(f"[SIM IPC] accept error: {e}")

    def recv_message(self) -> Optional[dict]:
        self.poll_accept()
        if self.client is None:
            return None

        # Check readability without blocking
        try:
            readable, _, exceptional = select.select([self.client], [], [self.client], 0.0)
        except (OSError, ValueError) as e:
            self._drop_client(f"select error: {e}")
            return None

        if exceptional:
            self._drop_client("socket exception flag")
            return None

        if not readable:
            return None

        # Drain all available data in a loop to handle partial sends
        try:
            while True:
                chunk = self.client.recv(1 << 20)  # 1 MB chunks
                if not chunk:
                    self._drop_client("client closed connection")
                    return None
                self.buffer += chunk
                # Check if more data is immediately available
                r, _, _ = select.select([self.client], [], [], 0.0)
                if not r:
                    break
        except BlockingIOError:
            pass  # No more data right now — normal
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self._drop_client(f"recv error: {e}")
            return None

        # Parse one complete newline-delimited JSON message
        if b"\n" not in self.buffer:
            return None

        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            return None

        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[SIM IPC] JSON decode error: {e} | raw={line[:120]}")
            return None

    def send_message(self, payload: dict) -> bool:
        if self.client is None:
            return False
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self.client.sendall(data)
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self._drop_client(f"send error: {e}")
            return False

    def close(self):
        self._drop_client()
        try:
            self.server.close()
        except Exception:
            pass


# ===============================================================
# IsaacSim Server
# ===============================================================
class IsaacSimServer:
    def __init__(self):
        self.stage = None
        self.timeline = None
        self.robot = None
        self.camera = None
        self.spawn_region = None
        self.occ_map = OccupancyDistanceMap(
            "/nas/sujinkim/data/goto/sim/goto_warehouse.png",
            "/nas/sujinkim/data/goto/sim/goto_warehouse.yaml",
        )

        self.obs_prim = None
        self.obs_translate_op = None
        self.obs_base_y = -5.2
        self.obs_base_z = 0.5
        self.obs_center_x = -12.0
        self.obs_amplitude = 4.0
        self.obs_speed = 0.6
        self.obs_phase = 0.0

    def setup(self):
        self._enable_extensions()
        self._create_stage_once()
        self._load_env_and_robot_once()
        self._start_simulation_once()
        self._initialize_articulation_once()
        self._setup_camera_once()
        self._setup_dynamic_obstacle_once()

    def _enable_extensions(self):
        enable_extension("omni.physx")
        enable_extension("omni.physx.ui")
        enable_extension("omni.graph.nodes")
        enable_extension("isaacsim.core.nodes")
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.rtx")
        simulation_app.update()
        simulation_app.update()

    def _create_stage_once(self):
        create_new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        add_physics_scene(self.stage)
        add_dome_light(self.stage)

    def _load_env_and_robot_once(self):
        assets_root = get_assets_root_path()
        robot_usd = assets_root + ROBOT_REL_PATH
        add_reference_to_stage(usd_path=ENV_USD_PATH, prim_path=ENV_PRIM_PATH)
        add_reference_to_stage(usd_path=robot_usd, prim_path=ROBOT_ROOT_PRIM_PATH)
        simulation_app.update()
        simulation_app.update()
        self.spawn_region = get_world_bbox_xy(self.stage, SPAWN_AREA_PRIM_PATH)

    def _start_simulation_once(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        for _ in range(20):
            simulation_app.update()

    def _initialize_articulation_once(self):
        self.robot = Articulation(ROBOT_ROOT_PRIM_PATH)
        self.robot.initialize()
        for _ in range(10):
            simulation_app.update()

    @staticmethod
    def yaw_to_quat_xyzw(yaw: float):
        half = yaw * 0.5
        return [0.0, 0.0, math.sin(half), math.cos(half)]

    @staticmethod
    def quat_multiply_xyzw(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    @staticmethod
    def euler_xyz_deg_to_quat_xyzw(rx_deg, ry_deg, rz_deg):
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)
        cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
        cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
        cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
        qw = cx * cy * cz - sx * sy * sz
        qx = sx * cy * cz + cx * sy * sz
        qy = cx * sy * cz - sx * cy * sz
        qz = cx * cy * sz + sx * sy * cz
        return [qx, qy, qz, qw]

    @staticmethod
    def npquat_xyzw_to_gf(q):
        x, y, z, w = q
        return Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))

    def set_xform_pose(self, prim, xyz, quat_xyzw):
        quatd = self.npquat_xyzw_to_gf(quat_xyzw)
        xformable = UsdGeom.Xformable(prim)
        ordered_ops = xformable.GetOrderedXformOps()
        translate_op = None
        orient_op = None
        for op in ordered_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        orient_op.Set(quatd)

    @staticmethod
    def fov_to_focal_length(fov_deg, aperture=20.955):
        fov_rad = math.radians(fov_deg)
        return aperture / (2.0 * math.tan(fov_rad / 2.0))

    def _setup_camera_once(self):
        stage = self.stage
        if not stage.GetPrimAtPath("/World/replay_camera").IsValid():
            UsdGeom.Xform.Define(stage, "/World/replay_camera")
        cam_prim_path = FRONT_CAM_CFG["camera_prim_path"]
        if not stage.GetPrimAtPath(cam_prim_path).IsValid():
            UsdGeom.Camera.Define(stage, cam_prim_path)

        self.camera = Camera(
            prim_path=cam_prim_path,
            name=FRONT_CAM_CFG["name"],
            frequency=30,
            resolution=FRONT_CAM_CFG["resolution"],
        )
        self.camera.initialize()

        cam_prim = stage.GetPrimAtPath(cam_prim_path)
        cam_geom = UsdGeom.Camera(cam_prim)
        cam_geom.GetHorizontalApertureAttr().Set(20.955)
        cam_geom.GetVerticalApertureAttr().Set(15.2908)
        cam_geom.GetFocalLengthAttr().Set(self.fov_to_focal_length(FRONT_CAM_CFG["fov_deg"]))

        for _ in range(10):
            simulation_app.update()

    def _setup_dynamic_obstacle_once(self):
        self.obs_prim = self.stage.GetPrimAtPath(DYNAMIC_OBSTACLE_PATH)
        if not self.obs_prim.IsValid():
            raise RuntimeError(f"Dynamic obstacle prim not found: {DYNAMIC_OBSTACLE_PATH}")

        xformable = UsdGeom.Xformable(self.obs_prim)
        self.obs_translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                self.obs_translate_op = op
                break
        if self.obs_translate_op is None:
            self.obs_translate_op = xformable.AddTranslateOp()
        self.obs_translate_op.Set(
            Gf.Vec3d(self.obs_center_x, self.obs_base_y, self.obs_base_z)
        )

    def update_dynamic_obstacle(self, dt: float):
        if self.obs_translate_op is None:
            return
        self.obs_phase += self.obs_speed * dt
        x = self.obs_center_x + self.obs_amplitude * math.sin(self.obs_phase)
        self.obs_translate_op.Set(Gf.Vec3d(x, self.obs_base_y, self.obs_base_z))

    def compose_camera_world_pose(self, robot_pose_world: Tuple[float, float, float]):
        base_x, base_y, base_yaw = robot_pose_world
        dx, dy, dz = FRONT_CAM_CFG["offset_xyz"]
        r_deg, p_deg, y_deg = FRONT_CAM_CFG["rot_xyz_deg"]
        cam_x = base_x + math.cos(base_yaw) * dx - math.sin(base_yaw) * dy
        cam_y = base_y + math.sin(base_yaw) * dx + math.cos(base_yaw) * dy
        cam_z = dz
        q_base = self.yaw_to_quat_xyzw(base_yaw)
        q_cam_local = self.euler_xyz_deg_to_quat_xyzw(r_deg, p_deg, y_deg)
        q_cam_world = self.quat_multiply_xyzw(q_base, q_cam_local)
        return [cam_x, cam_y, cam_z], q_cam_world

    def reset_robot_random_pose(self):
        x, y, yaw = sample_conditioned_spawn(self.spawn_region, self.occ_map)
        quat_wxyz = quat_wxyz_from_yaw(yaw)
        self.robot.set_world_pose(
            position=np.array([x, y, DEFAULT_Z], dtype=np.float32),
            orientation=quat_wxyz,
        )
        self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        for _ in range(120):
            simulation_app.update()
        return self.get_pose()

    def get_pose(self):
        x, y, yaw = get_world_xy_yaw(self.stage, ROBOT_BODY_PRIM_PATH)
        return {"x": x, "y": y, "yaw": yaw}

    def get_rgb_base64(self):
        t0_total = now_perf()

        t0 = now_perf()
        pose = self.get_pose()
        t_pose = now_perf()
        push_server_metric("srv.pose_ms", (t_pose - t0) * 1000.0)

        t0 = now_perf()
        cam_xyz, cam_quat = self.compose_camera_world_pose(
            (pose["x"], pose["y"], pose["yaw"])
        )
        cam_prim = self.stage.GetPrimAtPath(FRONT_CAM_CFG["camera_prim_path"])
        self.set_xform_pose(cam_prim, cam_xyz, cam_quat)
        t_cam_pose = now_perf()
        push_server_metric("srv.cam_pose_ms", (t_cam_pose - t0) * 1000.0)

        t0 = now_perf()
        for _ in range(2):
            simulation_app.update()
        t_after_updates = now_perf()
        push_server_metric("srv.sim_update_ms", (t_after_updates - t0) * 1000.0)

        t0 = now_perf()
        rgba = self.camera.get_rgba()
        t_img_capture = now_perf()
        push_server_metric("srv.rgba_ms", (t_img_capture - t0) * 1000.0)

        obs_timestamp = time.time()

        t0 = now_perf()
        rgb = np.asarray(rgba)[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        t_encode_done = now_perf()
        push_server_metric("srv.encode_ms", (t_encode_done - t0) * 1000.0)
        push_server_metric("srv.obs_total_ms", (t_encode_done - t0_total) * 1000.0)

        return {
            "image_b64": image_b64,
            "timestamp": obs_timestamp,
            "timing": {
                "t_srv_pose": t_pose,
                "t_srv_img_capture": t_img_capture,
                "t_srv_encode_done": t_encode_done,
            },
            "latency_ms": {
                "pose_ms": (t_pose - t0_total) * 1000.0,
                "cam_pose_ms": server_metric_last("srv.cam_pose_ms"),
                "sim_update_ms": server_metric_last("srv.sim_update_ms"),
                "rgba_ms": server_metric_last("srv.rgba_ms"),
                "encode_ms": server_metric_last("srv.encode_ms"),
                "obs_total_ms": server_metric_last("srv.obs_total_ms"),
            },
        }


def main():
    server = JsonSocketServer(host="0.0.0.0", port=8765)
    sim = IsaacSimServer()
    sim.setup()

    print("[SIM SERVER] ready, listening on 0.0.0.0:8765")

    try:
        while simulation_app.is_running():
            try:
                simulation_app.update()
                sim.update_dynamic_obstacle(1.0 / 60.0)

                msg = server.recv_message()
                if msg is None:
                    continue

                cmd = msg.get("cmd")

                if cmd == "ping":
                    server.send_message({"ok": True, "msg": "pong"})

                elif cmd == "reset":
                    pose = sim.reset_robot_random_pose()
                    ok = server.send_message({"ok": True, "pose": pose})
                    if not ok:
                        print("[SIM SERVER] send failed for reset response")

                elif cmd == "get_obs":
                    global SERVER_OBS_SEQ
                    SERVER_OBS_SEQ += 1

                    t_req_recv = now_perf()
                    pose = sim.get_pose()
                    obs_packet = sim.get_rgb_base64()
                    t_send = now_perf()

                    payload = {
                        "ok": True,
                        "obs_id": SERVER_OBS_SEQ,
                        "pose": pose,
                        "image_b64": obs_packet["image_b64"],
                        "timestamp": obs_packet["timestamp"],
                        "timing": {
                            "t_srv_req_recv": t_req_recv,
                            "t_srv_pose": obs_packet["timing"]["t_srv_pose"],
                            "t_srv_img_capture": obs_packet["timing"]["t_srv_img_capture"],
                            "t_srv_encode_done": obs_packet["timing"]["t_srv_encode_done"],
                            "t_srv_send": t_send,
                        },
                        "server_latency_ms": obs_packet["latency_ms"],
                    }
                    ok = server.send_message(payload)
                    if not ok:
                        print(f"[SIM SERVER] send failed for get_obs #{SERVER_OBS_SEQ}")
                    else:
                        print(
                            f"[SERVER][OBS {SERVER_OBS_SEQ:06d}] "
                            f"ts={obs_packet['timestamp']:.6f} "
                            f"sensor_total={obs_packet['latency_ms']['obs_total_ms']:.1f}ms "
                            f"(rgba={obs_packet['latency_ms']['rgba_ms']:.1f}ms, "
                            f"enc={obs_packet['latency_ms']['encode_ms']:.1f}ms)"
                        )

                    if SERVER_OBS_SEQ % 50 == 0:
                        print_server_latency_summary()

                elif cmd == "get_pose":
                    pose = sim.get_pose()
                    server.send_message(
                        {"ok": True, "pose": pose, "timing": {"t_srv_pose_reply": now_perf()}}
                    )

                else:
                    server.send_message({"ok": False, "error": f"unknown cmd: {cmd}"})

            except Exception as e:
                print(f"[SIM SERVER] loop error: {e}")
                try:
                    server.send_message({"ok": False, "error": str(e)})
                except Exception:
                    pass

    finally:
        if sim.timeline is not None:
            sim.timeline.stop()
        server.close()
        simulation_app.close()


if __name__ == "__main__":
    main()