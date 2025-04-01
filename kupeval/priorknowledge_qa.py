import os
import sys
import pandas as pd
import random
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.gpt_engine import GPT_Engine
from codebase.knowledge_update.llm_engine import OpenLM_Engine
import argparse
from prompts import *

output_dir = '/share/goyal/lio/knowledge_delta/dataset'

"""
Generate QA for Prior Knowledge
"""
def generate_prior_qa (cache_filepath: str,
                       model = 'gpt-4o',
                       temperature = 0.7,
                       max_tokens = 512
                       ):
    prompt_template = '''You are provided with a factual statement. Your task is to convert the fact into a question answer pair. The question should be specific, unambiguous, and detailed. The answer that you provide should only contain a few important and necessary tokens, which will be used for exact match.
Format Requirement: The question should start with 'Question:', and the answer starts with 'Answer:'.

Below is an example
Fact: Alexander Nevsky Cathedral is the largest Eastern Orthodox cathedral in Sofia, Bulgaria.
Question: What is the largest Eastern Orthodox cathedral in Sofia, Bulgaria?  
Answer: Alexander Nevsky Cathedral

This is your task
Fact: {fact}
'''
    system_prompt = "You are a helpful assistant"
    
    input_df = pd.read_pickle(os.path.join(output_dir, 'alpha_dataset.pickle'))[['entity_id', 'fact']] \
                .drop_duplicates() \
                .reset_index(drop = True)

    template_map = {'fact': 'fact'}

    engine = GPT_Engine(input_df=input_df, \
                        prompt_template=prompt_template, \
                        developer_message=system_prompt, \
                        template_map=template_map,
                        nick_name='generate_prior_qa',
                        cache_filepath=cache_filepath,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        mode = 'chat_completions'
                        )
    engine._run_model(overwrite=True)

def format_qa (filepath: str):
    def parse(response: str):
        if 'Question' not in response or 'Answer' not in response:
            return None, None
        
        response = response.replace("*", "")
        try:
            question = response[:response.index("Answer")].replace("Question", "").replace(":", "").strip()
            answer = response[response.index("Answer"):].replace("Answer", "").replace(":", "").strip()
            return question, answer
        except:
            return None, None
        
    response_df = pd.read_pickle(filepath)
    response_df[["question", "ground_truth"]] = pd.DataFrame(response_df['response'].apply(lambda x: parse(x)).tolist())


    input_df = pd.read_pickle(os.path.join(output_dir, 'alpha_dataset.pickle'))[['entity_id', 'entity', 'fact']] \
                .drop_duplicates() \
                .reset_index(drop = True)
    
    output_df = input_df.merge(response_df, left_index = True, right_index = True)

    output_df.to_pickle(os.path.join(os.path.dirname(filepath), 'prior_knowledge_qa.pickle'))
    

"""
QA Evaluation Prior Knowledge
"""

class PriorKnowledge_QA (OpenLM_Engine):
    def __init__(self,
                filepath: str,
                model_name: str,
                output_dir: str,
                nick_name: str,
                max_tokens: int = 512,
                temperature: float = 1.0,
                n: int = 10,
                tf_question: bool = False
                ):
        self.output_dir = output_dir
        self.nick_name = nick_name
        self.n = n
        self.tf_question = tf_question

        print(f"Start running evals on {model_name}")
        # Setup dir and tokenizer

        tokenizer_name = self._get_tokenizer_name(model_name)

        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_tokens=max_tokens,
            temperature=temperature,
            input_prompts=None,
            n = self.n,
            stop=["###Question:", 'Question:']
        )

        self._load_df(filepath=filepath)
        self._get_prompt_template()
        self._prepare_input()
        self._run_eval()
        self._parse_response()
        self._save_results()

    def _get_tokenizer_name(self, model_name: str) -> str:
        model_name = model_name.lower()

        if "llama" in model_name:
            return "meta-llama/Llama-3.1-8B-Instruct"
        elif "mistral" in model_name:
            return "mistralai/Mistral-7B-Instruct-v0.3"
        else:
            raise ValueError("Model name not recognized")
    
    def _load_df (self, filepath: str):
        self.df = (
            pd.read_pickle(filepath)
            .dropna()
            .reset_index(drop=True)
        )
        # TODO: Change the in context example entity ids
        in_context_entity_ids = [4303, 5655, 5692, 5100]
        self.df = self.df[~self.df['entity_id'].isin(in_context_entity_ids)].reset_index(drop = True)

    def _get_prompt_template (self):
        if self.tf_question:
            self.prompt_template = '''Today's Date: December 31st, 2023
Your Knowledge Cutoff: December 2023

Question: Is the following statement about Serbian SuperLiga True, Partially True, or Completely False? Statement: The season of the Serbian SuperLiga typically runs from July to May, with a winter break in December and January.
Response: True. The Serbian SuperLiga season typically does run from July to May, and there is a winter break, usually lasting through late December and most of January due to harsh winter weather.

Question: Is the following statement about Dyson Supersonic Hair Dryer True, Partially True, or Completely False? Statement: The Dyson Supersonic Hair Dryer is available in various color options, including fuchsia and nickel.
Response: Partially True. The Dyson Supersonic Hair Dryer is available in various color options, but the specific colors may vary depending on the region and the model.

Question: Is the following statement about Ghost of Tsushima True, Partially True, or Completely False? Statement: Ghost of Tsushima is available on the Xbox.
Response: Completely False. It is exclusively available on the PlayStation 4 and PlayStation 5 consoles.

Question: Is the following statement about The Great British Bake Off True, Partially True, or Completely False? Statement: The Great British Bake Off is broadcast on Channel 4 in the United Kingdom.
Response: True.

Question: Is the following statement about {entity} True, Partially True, or Completely False? Statement: {fact}
Response:'''
        else:
            self.prompt_template = '''Today's Date: December 31st, 2023
Your Knowledge Cutoff: December 2023

###Question: When does the season of the Serbian SuperLiga typically run?
Answer: The Serbian SuperLiga typically runs from July to May

###Question: On which platforms is Ghost of Tsushima available?
Answer: Ghost of Tsushima is available on the PlayStation 4 and PlayStation 5 platforms.

###Question: On which channel is The Great British Bake Off broadcast in the United Kingdom?
Answer: The Great British Bake Off is broadcast on Channel 4 in the United Kingdom.

###Question: {question}
Answer:'''

    def _prepare_input(self) -> None:
        if self.tf_question:
            self.input_prompts = [
                self.prompt_template.format(
                    entity = row['entity'],
                    fact = row['fact']
                )
                for _, row in self.df.iterrows()
            ]
        else:
            self.input_prompts = [
                self.prompt_template.format(
                    question = row['question']
                )
                for _, row in self.df.iterrows()
            ]

    def _run_eval (self) -> None:
        self.response_df =self._complete()
        self.response_df['entity_id'] = self.df['entity_id'].repeat(self.n).values

    def _parse_response(self) -> None:
        self.response_df.rename(columns = {'response': 'model_answer'}, inplace = True)

    def _save_results(self) -> None:
        self.response_df.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}_results.pickle'))

def tf_evaluate (nick_name: str):
    print(f"Parse Answers for {nick_name}")
    path = "/share/goyal/lio/knowledge_delta/evaluation/prior_knowledge_qa"
    qa_df = pd.read_pickle(os.path.join(path, 'questions/prior_knowledge_qa.pickle'))
    response_df = pd.read_pickle(os.path.join(path, f'tf_answers/{nick_name}_results.pickle'))

    df = qa_df[['entity_id', 'fact', 'question', 'ground_truth']].merge(response_df[['entity_id', 'model_answer']], on = ['entity_id'])

    def classify (series):
        model_answer = series['model_answer'].lower().split("\n")[0]
        if 'true' in model_answer and 'false' not in model_answer:
            return True
        elif 'true' not in model_answer and 'false' in model_answer:
            return False
        else:
            return None
        
    df['TF'] = df.apply(classify, axis = 1)
    df.to_pickle(os.path.join(path, f'tf_answers/{nick_name}_results.pickle'))

def qa_evaluate (nick_name: str):
    path = "/share/goyal/lio/knowledge_delta/evaluation/prior_knowledge_qa"
    qa_df = pd.read_pickle(os.path.join(path, 'questions/prior_knowledge_qa.pickle'))
    response_df = pd.read_pickle(os.path.join(path, f'answers/{nick_name}_results.pickle'))

    df = qa_df[['entity_id', 'fact', 'question', 'ground_truth']].merge(response_df[['entity_id', 'model_answer']], on = ['entity_id'])

    def exact_match (series):
        ground_truth = series['ground_truth'].strip()
        model_answer = series['model_answer'].strip()
        
        strict_em = ground_truth in model_answer
        ground_truth_words = ground_truth.split(" ")
        relaxed_em = any(word in model_answer for word in ground_truth_words)

        return strict_em, relaxed_em
    
    df['strict'], df['relaxed'] = zip(*df.apply(exact_match, axis = 1))
    df.to_pickle(os.path.join(path, f'answers/{nick_name}_results.pickle'))


if __name__=='__main__':
    """
    T/F Generate Answers
    """
    # parser = argparse.ArgumentParser(description="Parsing inputs for prior knowledge qa")

    # parser.add_argument('--model_name', type=str, required=True)
    # parser.add_argument('--nick_name', type=str, required=True)
    # args = parser.parse_args()

    # filepath = "/share/goyal/lio/knowledge_delta/evaluation/prior_knowledge_qa/questions/prior_knowledge_qa.pickle"
    # output_dir = "/share/goyal/lio/knowledge_delta/evaluation/prior_knowledge_qa/tf_answers"

    # PriorKnowledge_QA(
    #     filepath=filepath,
    #     output_dir=output_dir,
    #     tf_question=True,
    #     **vars(args)
    # )

    """
    Parse Answers
    """
    nick_names = ["llama8b_base", "mistral7b_base",
                  "llama8b_cpt", "mistral7b_cpt",
                  "llama8b_cpt_2prior", "mistral7b_cpt_2prior",
                  "llama8b_cpt_rephrase", "mistral7b_cpt_rephrase"
                  ]
    
    for nick_name in nick_names:
        tf_evaluate(nick_name)

