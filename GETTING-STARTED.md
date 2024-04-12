
Instructions, as suggested by @Orion. (Consider creating a version
with text interspersed as Google Colab and link it here at the top.)

```
python3 -m pip install --user virtualenv
python3 -m virtualenv .llama-fast
source .llama-fast/bin/activate
git clone https://github.com/pytorch/torchat.git
cd llama-fast
git submodule sync
git submodule update --init

# If we need PyTorch nightlies
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
# Otherwise
# pip install torch torchvision

pip install sentencepiece huggingface_hub
# Eventually should be (when Dave has the PyPI packages)
# pip install sentencepiece huggingface_hub executorch
# I had some issues with the pytorch submodule not downloading from ExecuTorch - not sure why

# To download Llama 2 models, go to https://huggingface.co/meta-llama/Llama-2-7b and go through steps to obtain access.

# Once approved, login with
huggingface-cli login
# You will be asked for a token from https://huggingface.co/settings/tokens

# Set the model and paths for stories15M as an example to test things on desktop and mobile
MODEL_NAME=stories15M
MODEL_PATH=checkpoints/${MODEL_NAME}/stories15M.pt
MODEL_DIR=~/llama-fast-exports

# Could we make this stories15 instead?
export MODEL_DOWNLOAD=meta-llama/Llama-2-7b-chat-hf
./scripts/prepare.sh $MODEL_DOWNLOAD
python generate.py --compile --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --device {cuda,cpu,mps}

```