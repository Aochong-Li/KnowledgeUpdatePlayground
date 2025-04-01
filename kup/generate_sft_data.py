import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')
sys.path.insert(1 , '/home/al2644/research/codebase/knowledge_update/dataset_creation')

from gpt_engine import GPT_Engine
from prompts import *
output_dir = '/share/goyal/lio/knowledge_delta/dataset'
from typing import Optional

def generate_sft (cache_filepath: str,
                  model = 'gpt-4o',
                  temperature = 0.7,
                  max_tokens = 2048
                  ):
    import pdb; pdb.set_trace()
    prompt_template = GENERATE_SFT_INPUT_PROMPT_TEMPLATE
    system_prompt = SYSTEM_PROMPT

    input_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/dataset/alpha_dataset.pickle")
    template_qa = '''[{“content”: “this is a question that is self-contained”, “role”: “user“}, {“content”: “this is the answer. and if needed, use single quotation mark 'like this'”, “role”: “assistant”}]'''
    template_map = {'template_qa': template_qa, 'article': 'article'}
    # For each change, we sample 1 articles to avoid too much similar data
    input_df = input_df.groupby('entity_id', group_keys=False).sample(n = 1, random_state = 42).reset_index(drop = True)

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_sft_cont',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode='chat_completions'
                      )

def parse_sft_data (filepath: str):
    sft_df = pd.read_pickle(filepath)

    def remove_comment (response: str):
        try:
            l_bracket = response.index('[')
            r_bracket = response.rindex(']')

            response = response[l_bracket: r_bracket + 1]
            return eval(response)
        except:
            None

    def parse_json(response: str):
        response = response.replace('```', '').replace('json', '').replace("”", '"')
        try:
            return eval(response)
        except:
            retry = remove_comment(response)
            if retry is not None and isinstance(retry, list):
                return retry
            else:
                None

    def validate_chat_structure(chat: Optional[str]):
        """
        Mistral requires user/assistant/user... order.
        """
        if chat is None:
            return None
        
        roles = [msg['role'] for msg in chat]
        for i in range(1, len(roles)):
            if roles[i] == roles[i-1]:
                return None
        return chat
    
    import pdb; pdb.set_trace()
    sft_df['sft'] = sft_df['response'].apply(parse_json)
    sft_df['sft'] = sft_df['sft'].apply(validate_chat_structure)
    
    sft_df.to_pickle(os.path.join(output_dir, 'sft/alpha/sft_table.pickle'))
 
def generate_sft_dataset(num_train_per_category = 80):
    from datasets import Dataset, DatasetDict

    sft_df = pd.read_pickle(os.path.join(output_dir, 'sft/alpha/sft_table.pickle'))
    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_pool.pickle'))
    sft_df = entity_table[['entity_id', 'category']].merge(sft_df, on = 'entity_id')

    # Splitting train and test
    train_sft_df = sft_df[sft_df['sft'].notnull()]
    train_sft_df = train_sft_df.groupby('category', group_keys=False).sample(n=num_train_per_category)
    test_sft_df= sft_df[~sft_df['entity_id'].isin(train_sft_df['entity_id'])]


    train_dataset = Dataset.from_pandas(train_sft_df.reset_index(drop = True))
    test_dataset = Dataset.from_pandas(test_sft_df[test_sft_df['sft'].notnull()].reset_index(drop = True))

    ds = DatasetDict({"train_sft": train_dataset, "test_sft": test_dataset})

    train_sft_df.to_pickle(os.path.join(output_dir, 'sft/alpha_train_test/train_sft.pickle'))
    test_sft_df.to_pickle(os.path.join(output_dir, 'sft/alpha_train_test/test_sft.pickle'))

    ds.save_to_disk(os.path.join(output_dir, 'sft/alpha_train_test'))

if __name__=='__main__':
    generate_sft_dataset()
    # filepath = "/share/goyal/lio/knowledge_delta/dataset/sft/alpha/candidate/generate_sft_cont.pickle"
    # engine = generate_sft(cache_filepath=filepath)
    # engine._run_model(overwrite=True)

    # filepath = '/share/goyal/lio/knowledge_delta/dataset/sft/alpha/candidate/generate_sft.pickle'
    # parse_sft_data(filepath)

