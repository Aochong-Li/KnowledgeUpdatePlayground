import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.gpt_engine import GPT_Engine
from prompts import *

output_dir = '/share/goyal/lio/knowledge_delta/dataset'

def generate_update (cache_filepath: str,
                    model = 'gpt-4o',
                    temperature = 1.0,
                    max_tokens = 2048,
                    mode = "chat_completions"
                    ):
    prompt_template = STEP2_GENERATE_UPDATE_INPUT_PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT
    import pdb; pdb.set_trace()

    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_pool.pickle'))
    fact_table = pd.read_pickle(os.path.join(output_dir, 'update/alpha/remain_table.pickle'))
    input_df = entity_table.merge(fact_table, on = 'entity_id').reset_index(drop = True)

    template_map = {'entity': 'entity', 'category': 'category', 'fact': 'fact'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_updates_remain',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode=mode
                      )

def filter_update (input_filepath: str,
                   cache_filepath: str,
                   model: str,
                   temperature: float = 1.0,
                   max_tokens: int = 128,
                   n: int = 10
                   ):
    prompt_template = STEP2_FILTER_UPDATE_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP1_SYSTEM_PROMPT
    import pdb; pdb.set_trace()
    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_pool.pickle'))
    update_table  = pd.read_pickle(input_filepath)
    input_df = entity_table[['entity_id', 'entity', 'category']].merge(update_table, on = ['entity_id'])

    template_map = {'entity': 'entity', 'category': 'category', 'update': 'update'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='filter_updates',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      n = n)

def classify_update (cache_filepath: str,
                    model = 'gpt-4o',
                    temperature = 0.7,
                    max_tokens = 1024
                    ):
    import pdb; pdb.set_trace()
    prompt_template = STEP2_CLASSIFY_UPDATE_INPUT_PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT

    input_df = pd.read_pickle(os.path.join(output_dir, 'alpha_dataset.pickle'))[['entity_id', 'entity', 'fact', 'update']].drop_duplicates().reset_index(drop=True)

    template_map = {'entity': 'entity', 'fact': 'fact', 'update': 'update'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='classify_update',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )
    
# Parse Outputs
def parse_update(filepath: str):
    update_df = pd.read_pickle(filepath)
    
    def _parse (response: str):
        response = response.replace('*', '').replace('Update:\n', 'Update:')
        if 'Update:' not in response:
            return None
    
        split_index = response.rindex('Update:')
        update = response[split_index:].replace('Update:', '')
        if "not changeable" in update:
            return None
        else:
            return update
    
    update_df['update'] = update_df['response'].apply(_parse)
    update_df.to_pickle(filepath)

def parse_classify(filepath: str):
    import pdb; pdb.set_trace()
    classify_df = pd.read_pickle(filepath)

    def _parse (response: str):
        response = response.replace('*', '').replace('###', '')
        if 'Class:' not in response:
            return None
    
        split_index = response.rindex('Class:')
        update = response[split_index:].replace('Class:', '').strip().lower()

        if 'conceptual change' in update and 'entity substitution' not in update:
            return 'contetxual rewrite'
        elif 'entity substitution' in update and 'conceptual change' not in update:
            return 'entity substitution'
        else:
            return None
        
    classify_df['update type'] = classify_df['response'].apply(_parse)

    input_df = pd.read_pickle(os.path.join(output_dir, 'alpha_dataset.pickle'))[['entity_id', 'entity', 'fact', 'update']].drop_duplicates().reset_index(drop=True)

    classify_df = input_df.merge(classify_df, left_index = True, right_index = True)
    classify_df[['entity_id', 'response', 'update type']].to_pickle(filepath)
    
def parse_process(filepath: str, merge: bool, test: bool = True):
    process_df = pd.read_pickle(filepath)
    
    def _parse(response: str):
        response = response.replace('*', '').replace('Description:\n', 'Description:').replace('Negation:\n', 'Negation:')
        if 'Description:' not in response:
            return None, None
        
        split_index = response.rindex('Description:')
        process = response[split_index:].replace('Description:', '')
        
        negation_lines = [line for line in response.split('\n') if 'Negation' in line]
        if not negation_lines:
            return None, None
        
        negation = negation_lines[0].replace('#', '').replace('Negation:', '').strip()
        if not negation:
            negation = None
        
        return negation, process
    process_df['reversal'], process_df['process'] = zip(*process_df['response'].apply(_parse))
    process_df.to_pickle(filepath)

    if merge:
        suffix = 'test' if test else 'table'
        update_table = pd.read_pickle(os.path.join(output_dir, f'update/update_{suffix}.pickle'))
        process_df.index = update_table.index

        update_table = update_table.merge(process_df[['reversal', 'process']].dropna(), left_index = True, right_index = True)
        update_table.to_pickle(os.path.join(output_dir, f'update/update_{suffix}.pickle'))

if __name__=='__main__':
    cache_filepath = "/share/goyal/lio/knowledge_delta/dataset/update/alpha/candidate/classify_update.pickle"
    parse_classify(cache_filepath)
    # engine = classify_update(cache_filepath=cache_filepath)
    # engine._run_model(overwrite=True)


'''
def generate_process (cache_filepath: str,
                    model = 'gpt-4o',
                    temperature = 1.0,
                    max_tokens = 1024
                    ):
    import pdb; pdb.set_trace()
    prompt_template = '' #STEP2_GENERATE_PROCESS_INPUT_PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT

    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_table.pickle'))
    update_table = pd.read_pickle(os.path.join(output_dir, 'update/update_table.pickle'))

    input_df = entity_table.merge(update_table, on = 'entity_id')

    template_map = {'entity': 'entity', 'category': 'category', 'fact': 'fact', 'update': 'update'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_process',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      )    

'''
