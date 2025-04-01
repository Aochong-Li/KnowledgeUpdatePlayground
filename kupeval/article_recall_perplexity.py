import pandas as pd
import os
import sys
sys.path.insert(0 , '/home/al2644/research/')

import pickle
from tqdm import tqdm
import argparse

from codebase.knowledge_update.evaluation.prompts import *
from codebase.knowledge_update.llm_engine import OpenLM_Engine, Perplexity_Engine
from rouge_score import rouge_scorer

output_dir = '/share/goyal/lio/knowledge_delta/evaluation/article_recall/alpha'

class FactPerplexity(Perplexity_Engine):
    def __init__(self, 
                 model_name: str,
                 nick_name: str,
                 filepath: str = "/share/goyal/lio/knowledge_delta/alpha_dataset.pickle"
                 ):
        self.df = pd.read_pickle(filepath)[['entity_id', 'fact']].drop_duplicates()
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
            input_prompts = list(self.df['fact']),
            batch_size=8
        )
        print(f"Computing Fact Perplexity for {nick_name} on {filepath}")
        self.perplexities = self._compute_perplexity()
        self.df['perplexity'] = self.perplexities
        self.df.to_pickle(os.path.join('/share/goyal/lio/knowledge_delta/evaluation/perplexity/fact', f'{self.nick_name}.pickle'))

class Article_Perplexity (Perplexity_Engine):
    def __init__(self, 
                 filepath: str,
                 model_name: str,
                 nick_name: str
                 ):
        self.df = pd.read_pickle(filepath)[['entity_id', 'article']].drop_duplicates()
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
            input_prompts = list(self.df['article']),
            batch_size=8
        )
        print(f"Computing Training Data Perplexity for {nick_name} on {filepath}")
        self.perplexities = self._compute_perplexity()
        self.df['perplexity'] = self.perplexities
        self.df.to_pickle(os.path.join('/share/goyal/lio/knowledge_delta/evaluation/perplexity/article', f'{self.nick_name}.pickle'))

class Article_Recall (OpenLM_Engine):
    def __init__(self, 
                 filepath: str,
                 model_name: str,
                 nick_name: str,
                 max_tokens: int = 2048,
                 temperature: float = 0.5
                 ):
        self.df = pd.read_pickle(filepath)[['entity_id', 'article']].drop_duplicates()
        self.nick_name = nick_name
        
        if "llama" in model_name.lower():
            tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif "mistral" in model_name.lower():
            tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.3"
        else:
            raise Exception("Model name not recognized")
        
        self._prepare_input()
        
        super().__init__(
            model_name = model_name,
            tokenizer_name = tokenizer_name,
            input_prompts = list(self.df['input_prompts']),
            max_tokens = max_tokens,
            temperature = temperature
        )
        self.output_df = super()._complete()

    def _prepare_input(self):
        def _extract_paragraphs (article: str, min_len: int = 300):
            paragraphs = article.split('\n')
            for i in range(len(paragraphs)):
                input_prompt = '\n'.join(paragraphs[:i])
                if len(input_prompt) >= min_len:
                    return input_prompt
            if len(article) >= min_len:
                return article[:min_len]
            else:
                return None

        self.df['input_prompts'] = self.df['article'].apply(_extract_paragraphs)
        self.df = self.df[self.df['input_prompts'].notnull()]

    def _merge_outputs (self):
        assert len(self.df) == len(self.output_df)
        self.output_df.rename(columns = {'response': self.nick_name}, inplace=True)
        self.output_df.index = self.df.index

        self.df = self.df.merge(self.output_df, left_index = True, right_index = True)

    def _compute_rougescore (self): 
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.df[['r1', 'r2', 'rL']] = None

        for i in tqdm(range(len(self.df))):
            idx, article, input_prompt, reproduced_article = self.df.index[i], self.df['article'].iloc[i], self.df['input_prompts'].iloc[i], self.df[self.nick_name].iloc[i]
            # NOTE: We only measure the remaing part of the article that is not used as input
            x = article.replace(input_prompt, '')
            y = reproduced_article

            scores = scorer.score(x, y)

            self.df.loc[idx, ['r1', 'r2', 'rL']] = scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rouge2'].fmeasure
        
        self.df.rename(columns = {
            'r1': f'{self.nick_name}_r1',
            'r2': f'{self.nick_name}_r2',
            'rL': f'{self.nick_name}_rL'
            },
            inplace = True
        )

        self.df[[
            'entity_id',\
            'input_prompts',\
            self.nick_name,\
            f'{self.nick_name}_r1',\
            f'{self.nick_name}_r2',\
            f'{self.nick_name}_rL'
            ]].to_pickle(os.path.join(output_dir, f'{self.nick_name}.pickle'))
        
if __name__=='__main__':
    filepath = '/share/goyal/lio/knowledge_delta/dataset/alpha_dataset.pickle'

    parser = argparse.ArgumentParser(description="Parse Arguments for nick_name and model_name")
    parser.add_argument("--nick_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()
    FactPerplexity(filepath=filepath, **vars(args))

    
    # article_recall = Article_Recall(filepath=filepath, **vars(args))
    # article_recall._merge_outputs()
    # article_recall._compute_rougescore()

    '''Deprecated Function'''

    # def _parse_input (self, num_reasoning_steps: int = 5):
        # self._header = self.input_df[self.data_col].apply(lambda x: [line for line in x.split('\n') if len(line) > 0 and '*' in line])
        # self._header = self._header.apply(lambda x: '\n\n'.join(x)).to_frame().rename(columns = {self.data_col: 'header'})

        # self._body = self.input_df[self.data_col].apply(lambda x: [line for line in x.split('\n') if len(line) > 0 and '*' not in line])
        # self._body = self._body.apply(lambda x: '\n\n'.join(x)).to_frame().rename(columns = {self.data_col: 'body'})

        # '''format reasoning'''
        # # self.input_df['bulletpoints'] = self.input_df['reasoning'].str.replace('⇒', '')
        # # pattern = re.compile(r"step\s*\d", re.IGNORECASE)
        # # self.input_df['bulletpoints'] = self.input_df['bulletpoints'].apply(lambda x: re.split(pattern, x)[1:])
        # # self.input_df['bulletpoints'] = self.input_df['bulletpoints'].apply(lambda x: [sentence.replace('(', ' ').replace(')',' ') for sentence in x])

    # assert self.input_df['bulletpoints'].apply(lambda x: len(x) == num_reasoning_steps).sum() == len(self.input_df), 'Error: some rows in reasoning column cannot be parsed properly'
    
    # def chunk_article (self, df, col):
    #     df['passage'] = df[col].apply(lambda x: [passage for passage in x.split('\n') if len(passage.strip()) > 0])
    #     df = df.explode('passage').reset_index(drop = True)

    #     return df

    # def breakdown_fact (self, completion_col:str = 'replicated_article', model = 'gpt-4o', temperature: float = 0.5,
    #                      max_tokens: int = 2048, batch_size = 20, mode = 'batch_stream', batch_rate_limit = 10):
    #     input_prompt_template = ATOMIC_FACT_PROMPT_TEMPLATE
    #     system_prompt = SYSTEM_PROMPT
    #     df = pd.read_pickle(os.path.join(self.output_dir, self.output_filename))

    #     df = self.chunk_article(df, completion_col)
        
    #     template_attribute_name_map = {'entity': self.entity_col, 'passage': 'passage'}

    #     engine =  GPT_Engine(input_df=df, \
    #                     input_prompt_template=input_prompt_template, \
    #                     system_prompt=system_prompt, \
    #                     template_attribute_name_map=template_attribute_name_map,
    #                     nick_name='atomic_fact',
    #                     cache_filepath=os.path.join(self.output_dir, 'atomic_fact.pickle'),
    #                     model=model,
    #                     temperature=temperature,
    #                     max_tokens=max_tokens,
    #                     batch_size=batch_size,
    #                     mode=mode,
    #                     batch_rate_limit=batch_rate_limit
    #                     )

    #     engine._run_model()
    #     output_df = engine._retrieve_outputs()

    #     '''Merging with input_df'''
    #     output_df = df.merge(output_df.rename(columns={'response': 'atomic_fact'}), left_index = True, right_index = True)
    #     output_df.to_pickle(os.path.join(self.output_dir, 'atomic_fact.pickle'))

    # def verify_fact_with_wikipedia (self, article_col:str = 'text', model = 'gpt-4o', temperature: float = 0.7,
    #                                 max_tokens: int = 512, batch_size = 20, mode = 'batch_stream', batch_rate_limit = 12):
    #     input_prompt_template = VERIFY_FACTS_WITH_ARTICLE_PROMPT_TEMPLATE
    #     system_prompt = SYSTEM_PROMPT

    #     df = pd.read_pickle(os.path.join(self.output_dir, 'atomic_fact.pickle'))
    #     article_df = self.input_df[[self.entity_col, article_col]].drop_duplicates()
    #     df = df.merge(article_df, left_on = self.entity_col, right_on = self.entity_col)

    #     df['objective_fact'] = df['atomic_fact'].apply(lambda x: x.split('Subjective Facts')[0].replace('Objective ', ''))
    #     template_attribute_name_map = {'entity': self.entity_col, 'evidence': article_col, 'facts': 'objective_fact'}

    #     engine =  GPT_Engine(input_df=df, \
    #                     input_prompt_template=input_prompt_template, \
    #                     system_prompt=system_prompt, \
    #                     template_attribute_name_map=template_attribute_name_map,
    #                     nick_name='wiki_verify_fact',
    #                     cache_filepath=os.path.join(self.output_dir, 'wikipedia_verification.pickle'),
    #                     model=model,
    #                     temperature=temperature,
    #                     max_tokens=max_tokens,
    #                     batch_size=batch_size,
    #                     mode=mode,
    #                     batch_rate_limit=batch_rate_limit
    #                     )
    #     engine._run_model()

    #     return engine
    
    # def verify_fact_with_completed_article (self, article_col:str = 'article', model = 'gpt-4o', temperature: float = 0.5,
    #                                 max_tokens: int = 2048, batch_size = 20, mode = 'batch_stream', batch_rate_limit = 12):
    #     input_prompt_template = VERIFY_FACTS_WITH_ARTICLE_PROMPT_TEMPLATE
    #     system_prompt = SYSTEM_PROMPT

    #     df = pd.read_pickle(os.path.join(self.output_dir, 'atomic_fact.pickle'))

    #     df['objective_fact'] = df['atomic_fact'].apply(lambda x: x.split('Subjective Facts')[0].replace('Objective ', ''))
    #     template_attribute_name_map = {'entity': self.entity_col, 'evidence': article_col, 'facts': 'objective_fact'}

    #     engine =  GPT_Engine(input_df=df, \
    #                     input_prompt_template=input_prompt_template, \
    #                     system_prompt=system_prompt, \
    #                     template_attribute_name_map=template_attribute_name_map,
    #                     nick_name='completed_article_verify_fact',
    #                     cache_filepath=os.path.join(self.output_dir, 'completed_article_verification.pickle'),
    #                     model=model,
    #                     temperature=temperature,
    #                     max_tokens=max_tokens,
    #                     batch_size=batch_size,
    #                     mode=mode,
    #                     batch_rate_limit=batch_rate_limit
    #                     )
    #     engine._run_model()

    #     return engine
    
    # def verify_fact_with_header (self, model = 'gpt-4o', temperature: float = 0.5, max_tokens: int = 204,
    #                             batch_size = 20, mode = 'batch_stream', batch_rate_limit = 12):
    #     input_prompt_template = VERIFY_FACTS_WITH_TITLE_PROMPT_TEMPLATE
    #     system_prompt = SYSTEM_PROMPT

    #     df = pd.read_pickle(os.path.join(self.output_dir, 'atomic_fact.pickle'))

    #     df['objective_fact'] = df['atomic_fact'].apply(lambda x: x.split('Subjective Facts')[0].replace('Objective ', ''))
    #     header_df = df[self.data_col].apply(lambda x: [line.replace('*', '') for line in x.split('\n') \
    #                                                   if len(line) > 0 and '*' in line][-1])
    #     df['header'] = header_df

    #     template_attribute_name_map = {'entity': self.entity_col, 'header': 'header', 'facts': 'objective_fact'}

    #     engine =  GPT_Engine(input_df=df, \
    #                     input_prompt_template=input_prompt_template, \
    #                     system_prompt=system_prompt, \
    #                     template_attribute_name_map=template_attribute_name_map,
    #                     nick_name='header_verify_fact',
    #                     cache_filepath=os.path.join(self.output_dir, 'header_verification.pickle'),
    #                     model=model,
    #                     temperature=temperature,
    #                     max_tokens=max_tokens,
    #                     batch_size=batch_size,
    #                     mode=mode,
    #                     batch_rate_limit=batch_rate_limit
    #                     )
    #     engine._run_model()

    #     return engine
    
    # def verification (self):
    #     df = pd.read_pickle(os.path.join(self.output_dir, 'atomic_fact.pickle'))

    #     wiki_verify_df = pd.read_pickle(os.path.join(self.output_dir, 'wikipedia_verification.pickle'))
    #     article_verify_df = pd.read_pickle(os.path.join(self.output_dir, 'completed_article_verification.pickle'))
    #     header_verify_df = pd.read_pickle(os.path.join(self.output_dir, 'header_verification.pickle'))

    #     def parse_fact (line: str):
    #         return line.replace('-', '').replace('[Support]', '').replace('[Not Support]', '')
        
    #     def format_label (df: pd.DataFrame, verify_col:str, col:str = 'response'):
    #         df['response'] = df['response'].apply(lambda x: [x for x in x.split('\n') if '-' in x and ('[Support]' in x or '[Not Support]' in x)])
    #         df[verify_col] = df['response'].apply(lambda x: {parse_fact(line): True if '[Support]' in line else False for line in x})

    #         return df[verify_col]
        
    #     wiki_verify_df = format_label(wiki_verify_df, 'wiki_verify')
    #     article_verify_df = format_label(article_verify_df, 'article_verify')
    #     header_verify_df = format_label(header_verify_df, 'header_verify')

    #     self.labeled_df =  pd.concat([df, wiki_verify_df, article_verify_df, header_verify_df], axis = 1)

    #     '''Filter reponses that miss facts'''
        
    #     self.labeled_df = self.labeled_df[ (self.labeled_df['wiki_verify'].apply(lambda x: len(x)) == self.labeled_df['article_verify'].apply(lambda x: len(x)))
    #                                       & (self.labeled_df['wiki_verify'].apply(lambda x: len(x)) == self.labeled_df['header_verify'].apply(lambda x: len(x))) 
    #                                       ]
        
    #     self.labeled_df['verified_fact'] = self.labeled_df['wiki_verify'].apply(lambda x: list(x.keys()))
    #     self.labeled_df['wiki_verify'] = self.labeled_df['wiki_verify'].apply(lambda x: list(x.values()))
    #     self.labeled_df['article_verify'] = self.labeled_df['article_verify'].apply(lambda x: list(x.values()))
    #     self.labeled_df['header_verify'] = self.labeled_df['header_verify'].apply(lambda x: list(x.values()))

    #     self.labeled_df = self.labeled_df.explode(['verified_fact', 'wiki_verify', 'article_verify', 'header_verify']).reset_index(drop = True)

    #     return self.labeled_df

    # def summarize_knowledge (self):
    #     self.count_df = self.labeled_df.copy()

    #     self.count_df['new_knowledge_count'] = self.count_df['article_verify'] & ~(self.count_df['wiki_verify'] | self.count_df['header_verify'])
    #     self.count_df['old_knowledge_count'] = self.count_df['wiki_verify'] | self.count_df['header_verify']
    #     self.count_df['hallucination_count'] = ~(self.count_df['article_verify'] | self.count_df['wiki_verify'] | self.count_df['header_verify'])

    #     summ_df = self.count_df.groupby(['title', 'article', 'replicated_article'])[['new_knowledge_count', 'old_knowledge_count', 'hallucination_count']].sum().reset_index()

    #     return self.count_df, summ_df
    
    # def breakdown_bulletpoints (self, model = 'gpt-4o', temperature: float = 0.5, max_tokens: int = 512, batch_size = 20, mode = 'batch_stream', batch_rate_limit = 10):
    #     input_prompt_template = ATOMIC_FACT_PROMPT_TEMPLATE
    #     system_prompt = SYSTEM_PROMPT

    #     self.input_df['atomic_bulletpoints'] = self.input_df['bulletpoints'].apply(lambda x: ' '.join(x))
    #     template_attribute_name_map = {'entity': 'title', 'passage': 'atomic_bulletpoints'}

    #     engine =  GPT_Engine(input_df=self.input_df, \
    #             input_prompt_template=input_prompt_template, \
    #             system_prompt=system_prompt, \
    #             template_attribute_name_map=template_attribute_name_map,
    #             nick_name='atomic_bulletpoints',
    #             cache_filepath=os.path.join(self.output_dir, 'atomic_bulletpoints.pickle'),
    #             model=model,
    #             temperature=temperature,
    #             max_tokens=max_tokens,
    #             batch_size=batch_size,
    #             mode=mode,
    #             batch_rate_limit=batch_rate_limit
    #     )

    #     engine._run_model()
    #     output_df = engine._retrieve_outputs()

    #     '''Merging with input_df'''
    #     self.input_df['atomic_bulletpoints'] = output_df['response']
    #     self.input_df.to_pickle(os.path.join(self.output_dir, 'atomic_bulletpoints.pickle'))

    # def recall_with_completed_article (self, model = 'gpt-4o', temperature: float = 0.5, max_tokens: int = 1024,
    #                             batch_size = 20, mode = 'batch_stream', batch_rate_limit = 12):
    #     df = pd.read_pickle(os.path.join(self.output_dir, 'atomic_bulletpoints.pickle')).reset_index(drop = True)
    #     df['objective_bulletpoints'] = df['atomic_bulletpoints'].apply(lambda x: x.split('Subjective Facts')[0].split('Objective Facts:')[1])

    #     replicated_article_df = pd.read_pickle(os.path.join(self.output_dir, self.output_filename))
    #     common_cols = replicated_article_df.columns.intersection(df.columns)

    #     df = df.merge(replicated_article_df, on = list(common_cols))

    #     input_prompt_template = RECALL_KNOWLEDGE_PROMPT_TEMPLATE
    #     system_prompt = SYSTEM_PROMPT
    #     template_attribute_name_map = {'article': 'replicated_article', 'atomic_bulletpoints': 'objective_bulletpoints'}

    #     prefix = [x for x in self.output_dir.split('/') if len(x) > 0][-1]
    #     return GPT_Engine(input_df=df, \
    #             input_prompt_template=input_prompt_template, \
    #             system_prompt=system_prompt, \
    #             template_attribute_name_map=template_attribute_name_map,
    #             nick_name=f'{prefix}_recalled_by_article',
    #             cache_filepath=os.path.join(self.output_dir, 'recalled_by_article.pickle'),
    #             model=model,
    #             temperature=temperature,
    #             max_tokens=max_tokens,
    #             batch_size=batch_size,
    #             mode=mode,
    #             batch_rate_limit=batch_rate_limit
    #     )

    #     # time.sleep(60)
    #     # output_df = engine._retrieve_outputs()
    #     # df['bulletpoints_recall'] = output_df['response']
    #     # df.to_pickle(os.path.join(self.output_dir, 'recalled_by_article.pickle'))
    
    # def compute_recall_score (self, col: str = 'bulletpoints_recall'):
    #     df = pd.read_pickle(os.path.join(self.output_dir, 'recalled_by_article.pickle'))
    #     df['supported'] = df[col].apply(lambda x: sum([1 for sent in x.split('\n') if '[Support]' in sent]))
    #     df['num_bulletpoints'] = df[col].apply(lambda x: sum([1 for sent in x.split('\n') if '[Support]' in sent or '[Not Support]' in sent]))

    #     df['article_recall'] = df['support'] / df['num_bulletpoints']

    #     return df