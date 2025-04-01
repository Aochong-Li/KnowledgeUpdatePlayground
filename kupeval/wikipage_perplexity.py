import pandas as pd
import os
import sys
sys.path.insert(0 , '/home/al2644/research/')

import argparse

from codebase.knowledge_update.evaluation.prompts import *
from codebase.knowledge_update.llm_engine import Perplexity_Engine

class WikiPagePerplexity(Perplexity_Engine):
    def __init__(self, 
                 model_name: str,
                 nick_name: str,
                 filepath: str = "/share/goyal/lio/knowledge_delta/dataset/entity/entity_wikipage_df.pickle"
                 ):
        self.df = pd.read_pickle(filepath)[['entity_id', 'wiki_page']].drop_duplicates()
        self.nick_name = nick_name
        
        if "llama" in model_name.lower():
            tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif "mistral" in model_name.lower():
            tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"
        else:
            raise Exception("Model name not recognized")
                
        super().__init__(
            model_name = model_name,
            tokenizer_name = tokenizer_name,
            input_prompts = list(self.df['wiki_page']),
            batch_size=1
        )
        print(f"Computing Wikipedia Page Perplexity for {nick_name} on {filepath}")
        self.perplexities = self._compute_perplexity()
        self.df['perplexity'] = self.perplexities
        self.df.to_pickle(os.path.join('/share/goyal/lio/knowledge_delta/evaluation/perplexity/wikipage', f'{self.nick_name}.pickle'))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Parse Arguments for nick_name and model_name")
    parser.add_argument("--nick_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()
    WikiPagePerplexity(**vars(args))
    #     python codebase/knowledge_update/evaluation/wikipage_perplexity.py --nick_name "llama8b_base" --model_name "meta-llama/Llama-3.1-8B"
