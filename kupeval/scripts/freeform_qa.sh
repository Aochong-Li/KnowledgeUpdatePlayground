#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

declare -A model_names=(
    # ["meta-llama/Llama-3.1-8B"]="llama8b_base"
    # ["Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_2prior"
    ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_2prior"
    
    # ["Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_prior"
    # ["Llama_3.1_8B/knowledge-alphallama8b_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_rephrase"

    # ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_prior"
    # ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_rephrase"
)
# Loop through the dictionary and run the command for each pair
for model_name in "${!model_names[@]}"; do
    nick_name=${model_names[$model_name]}
    
    echo "Evaluations begin with $nick_name"

    python codebase/knowledge_update/evaluation/freeform_qa/freeform_qa.py --nick_name $nick_name --model_name $model_name

    echo "Finished eval for model: $model_name"
done

echo "All runs completed."