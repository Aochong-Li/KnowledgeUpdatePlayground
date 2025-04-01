#!/bin/bash
# setting up default hyperparameters
subsample_ratio=1.0 # change this parameter to run the scaling plot

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr) lr="$2"; shift 2 ;; # learning rate
        --rt) rt="$2"; shift 2 ;; # repeat time
        --rr) rr="$2"; shift 2;; # replay rate
        --epochs) epochs="$2"; shift 2 ;;
        --block_size) block_size="$2"; shift 2;;
        --bs) bs="$2"; shift 2 ;; # batch size
        --wd) wd="$2"; shift 2 ;; # weight decay
        --warmup) warmup="$2"; shift 2 ;;
        --task_name) task_name="$2"; shift 2 ;;
        --split_name) split_name="$2"; shift 2;;
        --subsample_ratio) subsample_ratio="$2"; shift 2 ;;
        --model_name) model_name="$2"; shift 2 ;;
        --run_eval) run_eval=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done
gpu_count=$(nvidia-smi -L | wc -l)
if [[ ${model_name##*/} == *checkpoint-* ]]; then
    model_dir="$(basename "$(dirname "$model_name")")"
    checkpoint_dir="$(basename "$model_name")"
    pretty_name="${model_dir}_${checkpoint_dir}"
else
    # Otherwise, just take whatever is after the last slash
    pretty_name="$(basename "$model_name")"
fi
# Remove all hyphens from the result
pretty_name=$(echo "$pretty_name" | sed 's/-/_/g')

grad_acc=64
eff_bs=$((2*bs*grad_acc)) # effective batch size is 128

if [ "$subsample_ratio" = "1.0" ]; then
    run_name="${task_name}-${split_name}-lr${lr}-rt${rt}-rr${rr}-epochs${epochs}-blocksize${block_size}-bs${eff_bs}-wd${wd}-warmup${warmup}-${pretty_name}"
else
    run_name="scaling-subsample_ratio${subsample_ratio}-${task_name}-${split_name}-lr${lr}-rt${rt}-rr${rr}-epochs${epochs}-blocksize${block_size}-bs${eff_bs}-wd${wd}-warmup${warmup}-${pretty_name}"
fi
echo "Running experiment with run name: $run_name"
output_dir="/share/goyal/lio/knowledge_delta/training/model/sft/${run_name}"

if echo "$model_name" | grep -qi "llama"; then
    fsdp_config='{"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}'
else
    fsdp_config='{"transformer_layer_cls_to_wrap": "MistralDecoderLayer"}'
fi

cd /home/al2644/research/codebase/knowledge_update/training
# Execute the training command with the specific hyperparameters
torchrun --nproc_per_node=$gpu_count  train.py \
    --model_name=$model_name \
    --block_size=$block_size \
    --per_device_train_batch_size=$bs \
    --per_device_eval_batch_size=$((2 * bs)) \
    --gradient_accumulation_steps=$grad_acc \
    --num_train_epochs=$epochs \
    --learning_rate=$lr \
    --repeat_time=$rt \
    --replay_rate=$rr \
    --subsample_ratio=$subsample_ratio \
    --overwrite_output_dir=True \
    --task_name=$task_name \
    --split_name=$split_name \
    --logging_steps=1 \
    --run_name=$run_name \
    --bf16=True \
    --output_dir=$output_dir \
    --weight_decay=$wd \
    --warmup_ratio=$warmup \
    --evaluation_strategy="no" \
    --save_strategy="epoch" \
    --save_total_limit=10\
    --lr_scheduler_type="cosine" \
    --log_level="info" \
    --fsdp="hybrid_shard auto_wrap" \
    --fsdp_config="$fsdp_config" \