
# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_dist_inference.sh

NGPU=${NGPU:-"2"}

# TODO: We need to decide how to log for inference.
# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0,1}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
pp_meta.py --model 8b31
