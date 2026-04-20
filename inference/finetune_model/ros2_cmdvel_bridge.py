import json
import select
import socket
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class JsonSocketServer:
    """
    Non-blocking TCP server for receiving cmd_vel commands from edge client.
    Handles partial recv and client reconnects gracefully.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
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
                print(f"[CMD BRIDGE] client dropped: {reason}")

    def poll_accept(self):
        if self.client is not None:
            return
        readable, _, _ = select.select([self.server], [], [], 0.0)
        if readable:
            try:
                conn, addr = self.server.accept()
                conn.setblocking(False)
                self.client = conn
                self.buffer = b""
                print(f"[CMD BRIDGE] client connected from {addr}")
            except OSError as e:
                print(f"[CMD BRIDGE] accept error: {e}")

    def recv_message(self) -> Optional[dict]:
        self.poll_accept()
        if self.client is None:
            return None

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

        # Drain all available data
        try:
            while True:
                chunk = self.client.recv(65536)
                if not chunk:
                    self._drop_client("client closed connection")
                    return None
                self.buffer += chunk
                r, _, _ = select.select([self.client], [], [], 0.0)
                if not r:
                    break
        except BlockingIOError:
            pass
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self._drop_client(f"recv error: {e}")
            return None

        if b"\n" not in self.buffer:
            return None

        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            return None

        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[CMD BRIDGE] JSON decode error: {e}")
            return None

    def close(self):
        self._drop_client()
        try:
            self.server.close()
        except Exception:
            pass


class CmdVelPublisher(Node):
    def __init__(self, topic_name: str = "/cmd_vel"):
        super().__init__("omnivla_cmdvel_bridge")
        self.pub = self.create_publisher(Twist, topic_name, 10)

    def publish_twist(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelPublisher("/cmd_vel")
    server = JsonSocketServer("0.0.0.0", 8766)

    print("[CMD BRIDGE] listening on 0.0.0.0:8766")

    try:
        while rclpy.ok():
            msg = server.recv_message()
            if msg is not None:
                linear  = msg.get("linear", 0.0)
                angular = msg.get("angular", 0.0)
                node.publish_twist(linear, angular)
                print(f"[CMD BRIDGE] /cmd_vel: v={linear:.3f} w={angular:.3f}")

            rclpy.spin_once(node, timeout_sec=0.005)

    finally:
        # Publish zero before shutdown
        try:
            node.publish_twist(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.0)
        except Exception:
            pass
        server.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()