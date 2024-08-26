export CUDA_VISIBLE_DEVICES=4,5,6,7
PORT=${1:-29501}
NGPU=${NGPU:-"4"}

torchrun --nproc-per-node=$NGPU --master_port=$PORT dist_run.py
