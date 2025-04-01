#!/bin/bash

cd /home/al2644/research/github_repos/MMLU/lm-evaluation-harness
MODEL_DIR="/share/goyal/lio/knowledge_delta/training/model"
export CUDA_VISIBLE_DEVICES=1

# For parallelization, change 1. CUDA_VISIBLE_DEVICES, 2. model_names 3. batch_size

declare -A model_names=(
    # ["mistral_7B_v0.3_base"]="mistralai/Mistral-7B-v0.3"
    # ["llama3.2-3B_base"]="meta-llama/Llama-3.2-3B"
    # ["llama3.1-8B_base"]="meta-llama/Llama-3.1-8B"

    # ["mistral_7B_v0.3_cpt"]="Mistral_7B_v0.3/knowledge-alphamistral_cpt-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"
    # ["mistral_7B_v0.3_cpt_prior"]="Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"
    # ["mistral_7B_v0.3_cpt_rephrase"]="Mistral_7B_v0.3/knowledge-alphamistral_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"
    
    ["llama3.1-8B_cpt"]="Llama_3.1_8B/knowledge-alphallama8b_cpt-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"    
    ["llama3.1-8B_cpt_prior"]="Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"
    ["llama3.1-8B_cpt_rephrase"]="Llama_3.1_8B/knowledge-alphallama8b_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"
)
gpu_count=$(nvidia-smi -L | wc -l)

# Loop through the dictionary and run the command for each pair
for nick_name in "${!model_names[@]}"; do
    model_name=${model_names[$nick_name]}
    
    if [[ $model_name == *llama* ]]; then
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct"
    else
        tokenizer_name="mistralai/Mistral-7B-Instruct-v0.3"
    fi

    output_dir="/share/goyal/lio/knowledge_delta/evaluation/mmlu/alpha/$nick_name"
    
    if [[ ! $nick_name == *base* ]]; then
        model_name="$MODEL_DIR/$model_name"
    fi

    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi
    
    echo "MMLU: $nick_name, model_name: $model_name"

    lm-eval \
    --model hf \
    --model_args pretrained=$model_name,tokenizer=$tokenizer_name \
    --num_fewshot 5 \
    --tasks mmlu \
    --device auto \
    --batch_size 32 \
    --output_path "$output_dir"


    echo "Finished eval for model: $model_name"
done

echo "All runs completed."