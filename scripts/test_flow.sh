export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
rm -r checkpoints/$MODEL_REPO
python3 scripts/download.py --repo-id $MODEL_REPO
python3 scripts/convert_hf_checkpoint.py --checkpoint-dir checkpoints/$MODEL_REPO
python3 generate.py --compile --checkpoint-path checkpoints/$MODEL_REPO/model.pth --max-new-tokens 100
