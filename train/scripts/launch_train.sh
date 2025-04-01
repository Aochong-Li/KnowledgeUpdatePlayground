#!/bin/bash

# THIS IS FOR SFT
declare -A model_splits=(
    # ["/share/goyal/lio/knowledge_delta/training/model/Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="sft-train-llama3"
    # ["/share/goyal/lio/knowledge_delta/training/model/Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="sft-train-mistral"
    ["/share/goyal/lio/knowledge_delta/training/model/Llama_3.1_8B/knowledge-alphallama8b_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="sft-train-llama3"
    ["/share/goyal/lio/knowledge_delta/training/model/Mistral_7B_v0.3/knowledge-alphamistral_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="sft-train-mistral"
)

Loop through the dictionary and run the command for each pair
for model in "${!model_splits[@]}"; do
    split_name=${model_splits[$model]}
    
    echo "Running SFT for model: $model with split: $split_name"

    codebase/knowledge_update/training/scripts/instructSFT.sh \
        --lr 1e-05 \
        --rt 1 \
        --rr 10 \
        --epochs 1 \
        --block_size 2048 \
        --bs 1 \
        --wd 0.01 \
        --warmup 0.05 \
        --task_name instruct \
        --model_name "$model" \
        --split_name "$split_name"

    echo "Finished training for model: $model with split: $split_name"
done

echo "All runs completed."

# THIS IS FOR CPT PRIOR RLEARNING

# declare -A model_splits=(
#     # ["meta-llama/Llama-3.1-8B"]="alphallama8b_cpt_rephrase"
#     ["meta-llama/Llama-3.1-8B"]="alphallama8b_cpt_prior"

#     # ["mistralai/Mistral-7B-v0.3"]="alphamistral_cpt_rephrase"
#     ["mistralai/Mistral-7B-v0.3"]="alphamistral_cpt_prior"
# )

# # Loop through the dictionary and run the command for each pair
# for model in "${!model_splits[@]}"; do
#     split_name=${model_splits[$model]}
    
#     echo "Running training for model: $model with split: $split_name"

#     codebase/knowledge_update/training/scripts/cpt_train.sh \
#         --lr 1e-05 \
#         --rt 2 \
#         --rr 0.01 \
#         --epochs 1 \
#         --block_size 2048 \
#         --bs 1 \
#         --wd 0.01 \
#         --warmup 0.05 \
#         --task_name knowledge \
#         --model_name "$model" \
#         --split_name "$split_name"

#     echo "Finished training for model: $model with split: $split_name"
# done

# echo "All runs completed."