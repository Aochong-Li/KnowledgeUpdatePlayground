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

def tokenize_list_with_prior (
        knowledge_list: List[str],
        priors: List[str],
        model: str) -> List[int]:
    
    random.shuffle(knowledge_list)
    random.shuffle(priors)

    tokenizer = get_tokenizer(model)
    all_ids = []
    token_types = []
    
    # we tokenize them separately
    print('tokenizing knowledge articles')

    for text in tqdm(knowledge_list):
        if text:
            ids = tokenizer.encode(text, add_special_tokens=True) # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id)

            type_labels = len(ids) * [KNOWLEDGE_ARTICLE_TOKEN_TYPE]

            all_ids.extend(ids)
            token_types.extend(type_labels)
    
    print('tokenizing prior texts')

    for text in tqdm(priors):
        if text:
            ids = tokenizer.encode(text, add_special_tokens=True)  # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id)

            type_labels = len(ids) * [PRIOR_TOKEN_TYPE]

            all_ids.extend(ids)
            token_types.extend(type_labels)

    return all_ids, token_types

def tokenize_list(text_list: List[str], model: str) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer(model)

    all_ids = []
    token_types = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text, add_special_tokens=True) # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id) # add the end of text token
            all_ids.extend(ids)

            type_labels = len(ids) * [NONTARGET_ARTICLE_TOKEN_TYPE]
            token_types.extend(type_labels)
            
    return all_ids, token_types

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

    df = pd.read_pickle(os.path.join(ROOT, f'dataset/priorlearning/beta_dataset_{model_name}.pickle'))
    knowledge_article = list(df['article'])
    priors = []
    priors.extend(list(df['front_prior']))
    priors.extend(list(df['back_prior']))

    nontarget_content = pd.read_pickle(os.path.join(ROOT, 'dataset/nontarget_article/nontarget_article_table.pickle'))['content']

    dir = os.path.join(ROOT, f'training/priorlearning_control_data/{model_nickname}')
    if not os.path.isdir(dir):
        os.mkdir(dir)


    all_ids, token_type = tokenize_list_with_prior(knowledge_article, priors, model)
    write_to_memmap_single(all_ids, dir, f'knowledge_article.bin')
    write_to_memmap_single(token_type, dir, f'knowledge_article_token_type.bin')
      
    all_ids, token_type = tokenize_list(nontarget_content, model)
    write_to_memmap_single(all_ids, dir, f'nontarget_content.bin')
    write_to_memmap_single(token_type, dir, f'nontarget_content_token_type.bin')

if __name__ == '__main__':
    # for model_name in constants.MODEL_LIST.keys():
    model_name = "llama3.1-8B-instruct"
    print(f'Tokenizing for {model_name}')
    tokenize_knowledge(model_name=model_name)