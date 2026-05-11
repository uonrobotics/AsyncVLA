#!/usr/bin/env bash

# Nav2 (AMCL + costmap) — 실 로봇용, use_sim_time:=false
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/local_setup.bash

ros2 launch carter_navigation carter_navigation.launch.py \
    use_sim_time:=false
