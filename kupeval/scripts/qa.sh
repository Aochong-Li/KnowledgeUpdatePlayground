#!/bin/bash

# MCQ 

# prior learning
# model_name="Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"
# python codebase/knowledge_update/evaluation/qa.py --nick_name "llama8b_cpt_prior" --model_name $model_name

# model_name="Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"
# python codebase/knowledge_update/evaluation/qa.py --nick_name "mistral7b_cpt_prior" --model_name $model_name

# rephrase
# model_name="Llama_3.1_8B/knowledge-alphallama8b_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"
# python codebase/knowledge_update/evaluation/qa.py --nick_name "llama8b_cpt_rephrase" --model_name $model_name

# model_name="Mistral_7B_v0.3/knowledge-alphamistral_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"
# python codebase/knowledge_update/evaluation/qa.py --nick_name "mistral7b_cpt_rephrase" --model_name $model_name

export CUDA_VISIBLE_DEVICES=0

# model_name="sft/instruct-sft-train-mistral-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphamistral_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Mistral_7B_v0.3"
# python codebase/knowledge_update/evaluation/qa.py --nick_name "mistral7b_sft" --model_name $model_name

# model_name="sft/instruct-sft-train-llama3-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphallama8b_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Llama_3.1_8B"
# python codebase/knowledge_update/evaluation/qa.py --nick_name "llama8b_sft" --model_name $model_name


# Temporal MCQ
declare -A model_splits=(
    # ["meta-llama/Llama-3.1-8B"]="llama8b_base"
    # ["Llama_3.1_8B/knowledge-alphallama8b_cpt-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt"
    # ["Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_prior"
    # ["Llama_3.1_8B/knowledge-alphallama8b_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_rephrase"
    # ["sft/instruct-sft-train-llama3-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphallama8b_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Llama_3.1_8B"]="llama8b_sft"
    ["Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_2prior"

    # ["mistralai/Mistral-7B-v0.3"]="mistral7b_base"
    # ["Mistral_7B_v0.3/knowledge-alphamistral_cpt-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt"
    # ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_prior"
    # ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_rephrase"
    # ["sft/instruct-sft-train-mistral-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphamistral_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Mistral_7B_v0.3"]="mistral7b_sft"
    ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_2prior"
)

# # Loop through the dictionary and run the command for each pair
for model_name in "${!model_splits[@]}"; do
    nick_name=${model_splits[$model_name]}
    
    echo "Evaluating model: $model_name"

    python codebase/knowledge_update/evaluation/qa.py --nick_name $nick_name --model_name $model_name
    # python codebase/knowledge_update/evaluation/temporal_mcq.py --nick_name $nick_name --model_name $model_name

done