#!/usr/bin/env bash
set -e

# conda 영향 제거
unset PYTHONPATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

source /opt/ros/jazzy/setup.bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash

/usr/bin/python3.12 /home/sujin/workspace/physical-ai/AsyncVLA/inference/finetune_model/ros2_cmdvel_bridge.py