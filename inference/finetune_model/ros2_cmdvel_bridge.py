import json
import select
import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class JsonSocketServer:
    def __init__(self, host="0.0.0.0", port=8766):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.setblocking(False)

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
            print(f"[CMD BRIDGE] client connected from {addr}")

    def recv_message(self):
        self.poll_accept()
        if self.client is None:
            return None

        readable, _, _ = select.select([self.client], [], [], 0.0)
        if not readable:
            return None

        data = self.client.recv(65536)
        if not data:
            print("[CMD BRIDGE] client disconnected")
            self.client.close()
            self.client = None
            self.buffer = b""
            return None

        self.buffer += data
        if b"\n" not in self.buffer:
            return None

        line, self.buffer = self.buffer.split(b"\n", 1)
        if not line.strip():
            return None

        return json.loads(line.decode("utf-8"))

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        try:
            self.server.close()
        except Exception:
            pass


class CmdVelPublisher(Node):
    def __init__(self, topic_name="/cmd_vel"):
        super().__init__("omnivla_cmdvel_bridge")
        self.pub = self.create_publisher(Twist, topic_name, 10)

    def publish_twist(self, linear, angular):
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
                linear = msg.get("linear", 0.0)
                angular = msg.get("angular", 0.0)
                node.publish_twist(linear, angular)
                print(f"[CMD BRIDGE] publish /cmd_vel: v={linear:.3f}, w={angular:.3f}")

            rclpy.spin_once(node, timeout_sec=0.01)

    finally:
        node.publish_twist(0.0, 0.0)
        rclpy.spin_once(node, timeout_sec=0.0)
        server.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()