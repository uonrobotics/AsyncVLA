#!/usr/bin/env bash

set -e

# Using UV environment
source /home/sujin/workspace/physical-ai/AsyncVLA/.venv/bin/activate

python /home/sujin/workspace/physical-ai/AsyncVLA/inference/finetune_model/asyncvla_edge_client.py