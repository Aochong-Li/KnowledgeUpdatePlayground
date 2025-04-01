CACHED_MODELS = {
    'base-model-repeat': 'new_knowledge-newhandpicked_explicitnews-lr5e-06-rt10-rr0.1-epochs1-bs16-wd0.01-warmup0.05-Llama3.18B',
    'instruct-model-repeat': 'scaling-subsample_ratio0.25-instruct-ultrachat-train-lr1e-06-rt1-rr0.1-epochs1-bs128-wd0.01-warmup0.05-new_knowledgenewhandpicked_explicitnewslr5e06rt10rr0.1epochs1bs16wd0.01warmup0.05Llama3.18B/checkpoint-{checkpoint}',
    
    'base-model-rephrased5news': 'new_knowledge-newhandpicked_rephrased5news-lr5e-06-rt1-rr0.1-epochs5-bs16-wd0.01-warmup0.05-Llama3.18B/checkpoint-{checkpoint}',
    'base-model-rephrased5news-pl': 'new_knowledge-newhandpicked_rephrased5news_priorlearning-lr5e-06-rt1-rr0.1-epochs5-blocksize2048-bs16-wd0.01-warmup0.05-Llama3.18B/checkpoint-{checkpoint}',
    'sft-model-rephrased5news': '/share/goyal/lio/knowledge_update/continued_pretraining/model/instruct-explicitnews_sft-train-lr1e-06-rt1-rr0.9-epochs3-bs32-wd0.01-warmup0.05-new_knowledgenewhandpicked_rephrased5newslr5e06rt1rr0.1epochs5bs16wd0.01warmup0.05Llama3.18B_checkpoint614/checkpoint-{checkpoint}',

    'base-model-augment': 'new_knowledge-newhandpicked_augmenteddata-lr5e-06-rt1-rr0.1-epochs1-bs16-wd0.01-warmup0.05-Llama3.18B',
    'sft-model-augment': 'instruct-explicitnews_sft-train-lr1e-05-rt1-rr0.1-epochs10-bs8-wd0.01-warmup0.05-new_knowledgenewhandpicked_augmenteddatalr5e06rt1rr0.1epochs1bs16wd0.01warmup0.05Llama3.18B/checkpoint-{checkpoint}',
    
    'instruct-model-augment': 'scaling-subsample_ratio0.25-instruct-ultrachat-train-lr1e-06-rt1-rr0.1-epochs1-bs128-wd0.01-warmup0.05-new_knowledgenewhandpicked_augmenteddatalr5e06rt1rr0.1epochs1bs16wd0.01warmup0.05Llama3.18B/checkpoint-{checkpoint}'

}