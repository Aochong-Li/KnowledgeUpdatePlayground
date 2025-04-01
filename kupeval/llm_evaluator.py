import pandas as pd
import pickle
from tqdm import tqdm
import logging

from codebase.knowledge_update.llm_engine import OpenLM_Engine
from codebase.knowledge_update.evaluation.prompts import SYSTEM_PROMPT

class Evaluator(OpenLM_Engine):
    def __init__(self,
                 df: pd.DataFrame,
                 prompt_template: str = "",
                 system_prompt: str = SYSTEM_PROMPT,
                 template_map: dict = {},
                 model_name: str = "llama3.1-8B",
                 tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 n: int = 1,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.7,
                 top_k: int = 50,
                 logprobs: int = None,
                 prompt_logprobs: int = None
                 ):
        self.set_df(df)
        self.set_prompt_template(prompt_template)
        self.set_template_map(template_map)
        self.system_prompt = system_prompt

        self._prepare_prompts()

        if "llama" in model_name.lower():
            tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif "mistral" in model_name.lower():
            tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"


        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            input_prompts=self.input_prompts,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs
        )

    def set_df (self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def set_prompt_template (self, prompt_template: str):
        self.prompt_template = prompt_template

    def set_template_map (self, template_map: dict):
        self.template_map = template_map

    def _prepare_prompts(self) -> pd.Series:

        def format_prompt(row):
            properties = {
                k: row[v] if v in row.index else v
                for k, v in self.template_map.items()
            }
            return self.prompt_template.format(**properties)

        # Vectorized prompt creation using df.apply
        self.input_prompts = self.df.apply(format_prompt, axis=1)
        
        print(f"Generated {len(self.input_prompts)} input prompts.")