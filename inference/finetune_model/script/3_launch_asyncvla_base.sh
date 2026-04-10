#!/usr/bin/env bash

#*** [ 192.168.0.154 ] ***#

set -e

# Using UV environment
source /home/uon/sujinkim/AsyncVLA/.venv/bin/activate

python /home/uon/sujinkim/AsyncVLA/inference/finetune_model/asyncvla_base_server.py