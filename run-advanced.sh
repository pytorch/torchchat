set -eou pipefail
TORCHCHAT_ROOT=${PWD} ./scripts/install_et.sh
if false; then
python3 torchchat.py  ...  --help
fi
MODEL_NAME=stories15M
MODEL_DIR=~/checkpoints/${MODEL_NAME}
MODEL_PATH=${MODEL_DIR}/stories15M.pt
MODEL_OUT=~/torchchat-exports

mkdir -p ${MODEL_DIR}
mkdir -p ${MODEL_OUT}
python3 generate.py  --checkpoint-path ${MODEL_PATH} --prompt "Hello, my name is" --device  mps 
python3 export.py --checkpoint-path ${MODEL_PATH} --device  cuda  --output-pte-path ${MODEL_NAME}.pte
python3 export.py --checkpoint-path ${MODEL_PATH} --device  cpu  --output-dso-path ${MODEL_NAME}.so
python3 generate.py --checkpoint-path ${MODEL_PATH} --pte-path ${MODEL_NAME}.pte --device cpu --prompt "Once upon a time"
python3 generate.py --device {cuda,cpu} --dso-path ${MODEL_NAME}.so --prompt "Once upon a time"
if false; then
python3 generate.py --dtype  fp32 ...
python3 export.py --dtype  fp32 ...
fi
if false; then
python3  ...  --gguf-path <gguf_filename>
fi
exit 0
