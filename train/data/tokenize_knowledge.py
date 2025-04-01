import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append('/home/al2644/research/')
from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
import glob
from tqdm import tqdm
import pandas as pd
import pickle

from codebase.knowledge_update import constants

ROOT = '/share/goyal/lio/knowledge_delta'

KNOWLEDGE_ARTICLE_TOKEN_TYPE = 0
NONTARGET_ARTICLE_TOKEN_TYPE = 1
PRIOR_TOKEN_TYPE = 2

def get_tokenizer(tokenizer_model_name: str)-> AutoTokenizer:
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning
    return tokenizer

def tokenize_list(text_list: List[str], model: str) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer(model)

    all_ids = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text, add_special_tokens=True) # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id) # add the end of text token
            all_ids.extend(ids)
    
    return all_ids

def write_to_memmap_single(ids: List[int], dir: str, filename: str):
    if not os.path.isdir(os.path.join(dir, 'bins')):
        os.mkdir(os.path.join(dir, 'bins'))

    filename = f'bins/{filename}'
    filename = os.path.join(dir, filename)
    print(f'Writing to {filename} with length {len(ids)}')

    with open(os.path.join(dir, 'summary.txt'), 'a') as file:
        file.write(f'{filename}: {len(ids)}\n')

    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()

def tokenize_knowledge(model_name: str):
    model = constants.MODEL_LIST[model_name]
    model_nickname = model.split('/')[1]

    import pdb; pdb.set_trace()

    knowledge_article = pd.read_pickle(os.path.join(ROOT, 'dataset/alpha_dataset.pickle'))['article']
    nontarget_content = pd.read_pickle(os.path.join(ROOT, 'dataset/nontarget_article/alpha/nontarget_article_table.pickle'))['content']

    # Include Rephrase Data
    rephrase_data = pd.read_pickle(os.path.join(ROOT, 'dataset/rephrase/rephrase_table.pickle'))['rephrase']
    knowledge_article = pd.concat([rephrase_data, knowledge_article], ignore_index=True)

    dir = os.path.join(ROOT, f'training/rephrase/{model_nickname}')
    if not os.path.isdir(dir):
        os.mkdir(dir)
    
    write_to_memmap_single(tokenize_list(knowledge_article, model), dir, f'knowledge_article.bin')
    write_to_memmap_single(tokenize_list(nontarget_content, model), dir, f'nontarget_content.bin')

if __name__ == '__main__':
    # tokenize_knowledge(model_name='mistral-7b-instruct-v0.3')
    for model_name in constants.MODEL_LIST.keys():
        if 'instruct' not in model_name:
            continue
        print(f'Tokenizing for {model_name}')
        tokenize_knowledge(model_name=model_name)