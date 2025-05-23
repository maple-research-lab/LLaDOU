export PROC_PER_NODES=8

torchrun \
    --standalone \
    --nproc-per-node=$PROC_PER_NODES \
    --master-port=23443 \
    math_metrics.py \
        --ckpt_path models/LLaDOU-Math-8B \
        --local_data_path datasets/gsm8k \
        --num_workers 4 \
        --seed 112 \