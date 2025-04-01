declare -A model_splits=(
    # ["meta-llama/Llama-3.1-8B-Instruct"]="llama8b_rag"
    # ["sft/instruct-sft-train-llama3-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphallama8b_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Llama_3.1_8B"]="llama8b_sft"
   
    # ["sft/instruct-sft-train-mistral-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphamistral_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Mistral_7B_v0.3"]="mistral7b_sft"
    # ["sft/instruct-sft-train-llama3-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphallama8b_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Llama_3.1_8B"]="llama8b_cpt_sft"
    # ["sft/instruct-sft-train-mistral-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphamistral_cpt_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Mistral_7B_v0.3"]="mistral7b_cpt_sft"

    ["sft/instruct-sft-train-llama3-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphallama8b_cpt_rephrase_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Llama_3.1_8B"]="llama8b_cpt_rephrase_sft"
    ["sft/instruct-sft-train-mistral-lr1e-05-rt1-rr10-epochs1-blocksize2048-bs128-wd0.01-warmup0.05-knowledge_alphamistral_cpt_rephrase_lr1e_05_rt1_rr0.01_epochs1_blocksize2048_bs16_wd0.01_warmup0.05_Mistral_7B_v0.3"]="mistral7b_cpt_rephrase_sft"

)   

# # Loop through the dictionary and run the command for each pair
for model_name in "${!model_splits[@]}"; do
    nick_name=${model_splits[$model_name]}
    
    echo "Evaluating IndirectQA on model: $model_name"
    python codebase/knowledge_update/evaluation/indirect_probing.py --nick_name $nick_name --model_name $model_name

done