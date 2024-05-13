# Fine-tuned models from torchtune

<!--
[shell default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}" huggingface-cli login
[comment default]: TORCHCHAT_ROOT=${PWD} ./scripts/install_et.sh
-->

torchchat supports running inference with models fine-tuned using
[torchtune](https://github.com/pytorch/torchtune). To do so, we first
need to convert the checkpoints into a format supported by torchchat.

Below is a simple workflow to run inference on a fine-tuned Llama3
model. For more details on how to fine-tune Llama3, see the
instructions
[here](https://github.com/pytorch/torchtune?tab=readme-ov-file#llama3)

Start by installing torchtune:
```bash
# install torchtune
pip install torchtune
```

Download the model checkpoint with torchtune:

[shell default]: export HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}

[prefix default]: HF_TOKEN="${SECRET_HF_TOKEN_PERIODIC}
```
# download the llama3 model
tune download meta-llama/Meta-Llama-3-8B \
    --output-dir ./Meta-Llama-3-8B \
    --hf-token ${HF_TOKEN}
```

After installing torchtune and downloading the weights of the Llama3
model checkpoint to be torchtuned, we perform LoRA fine-tuning:

```
# Run LoRA fine-tuning on a single device. This assumes the config points to <checkpoint_dir> above
tune run lora_finetune_single_device --config llama3/8B_lora_single_device

# convert the fine-tuned checkpoint to a format compatible with torchchat
python3 build/convert_torchtune_checkpoint.py \
  --checkpoint-dir ./Meta-Llama-3-8B \
  --checkpoint-files meta_model_0.pt \
  --model-name llama3_8B \
  --checkpoint-format meta

# run inference on a single GPU
python3 torchchat.py generate \
  --checkpoint-path ./Meta-Llama-3-8B/model.pth \
  --device cuda
```

[end default]: end