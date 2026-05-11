#!/usr/bin/env bash

set -e

# ROS2 Jazzy (Python 3.12) — rclpy, cv_bridge, tf2_ros 등
source /opt/ros/jazzy/setup.bash

# Python 3.12 기반 edge venv (torch, torchvision 등)
source /home/sujin/workspace/physical-ai/AsyncVLA/.venv_edge/bin/activate

# prismatic 패키지 (소스 직접 참조)
export PYTHONPATH="/home/sujin/workspace/physical-ai/AsyncVLA:$PYTHONPATH"

python /home/sujin/workspace/physical-ai/AsyncVLA/inference/finetune_model/asyncvla_edge_client.py