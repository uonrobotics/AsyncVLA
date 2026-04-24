# Setup Instructions

## Set Up Conda Environment

<!-- ```bash
# Create and activate conda environment
conda create -n asyncvla python=3.10 -y
conda activate asyncvla

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/NHirose/AsyncVLA.git
cd AsyncVLA
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja -> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
``` -->

```bash
# UV environment
uv python install 3.10
uv venv --python 3.10
source .venv/bin/activate

uv pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  numpy==1.26.4 \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

uv pip install -e .

uv pip install packaging ninja
ninja --version
uv pip install --no-build-isolation "flash-attn==2.5.5"

```
