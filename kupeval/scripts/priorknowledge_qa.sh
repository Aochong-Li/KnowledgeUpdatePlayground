MODEL_PATH="/share/goyal/lio/knowledge_delta/training/model"

declare -A model_splits=(
    # ["meta-llama/Llama-3.1-8B"]="llama8b_base"
    # ["mistralai/Mistral-7B-v0.3"]="mistral7b_base"

   ["Llama_3.1_8B/knowledge-alphallama8b_cpt-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt"
   ["Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_2prior"
   ["Llama_3.1_8B/knowledge-alphallama8b_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"]="llama8b_cpt_rephrase"
   
   ["Mistral_7B_v0.3/knowledge-alphamistral_cpt-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt"
   ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_2prior"
   ["Mistral_7B_v0.3/knowledge-alphamistral_cpt_rephrase-lr1e-05-rt1-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Mistral_7B_v0.3"]="mistral7b_cpt_rephrase"
)   

# # Loop through the dictionary and run the command for each pair
for model_name in "${!model_splits[@]}"; do
    nick_name=${model_splits[$model_name]}
    
    echo "Evaluating PriorKnowledgeQA on model: $model_name"
    python codebase/knowledge_update/evaluation/priorknowledge_qa.py --nick_name $nick_name --model_name "$MODEL_PATH/$model_name"
done