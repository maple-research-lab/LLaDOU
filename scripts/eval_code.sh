export PROC_PER_NODES=8

torchrun \
    --standalone \
    --nproc-per-node=$PROC_PER_NODES \
    --master-port=23443 \
    code_metrics.py \
        --ckpt_path models/LLaDOU-v0-Code \
        --task MBPP \
        --seed 112 \