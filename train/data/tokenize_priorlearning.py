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

def tokenize_list_with_frontprior (
        knowledge_list: List[str],
        front_priors: List[str],
        back_priors: List[str],
        model: str):
    
    tuple_list = list(zip(front_priors, knowledge_list, back_priors))

    random.shuffle(tuple_list)
    tokenizer = get_tokenizer(model)
    all_ids = []
    token_types = []

    for data in tqdm(tuple_list):
        if data:
            front_prior = data[0]
            back_prior = data[2]
            knowledge = data[1]
            
            front_encoded = tokenizer.encode(front_prior, add_special_tokens=True)
            back_encoded = tokenizer.encode(back_prior, add_special_tokens=False)
            knowledge_encoded = tokenizer.encode(knowledge, add_special_tokens=False)

            # TODO: we put all priors infront of new knowledge
            ids = front_encoded + back_encoded + knowledge_encoded
            ids.append(tokenizer.eos_token_id)

            front_length = len(front_encoded)
            knowledge_length = len(knowledge_encoded)
            back_length = len(back_encoded)

            type_labels = (
                [PRIOR_TOKEN_TYPE] * (front_length + back_length) +
                [KNOWLEDGE_ARTICLE_TOKEN_TYPE] * (knowledge_length + 1) 
                )
            
            assert len(ids) == len(type_labels)

            all_ids.extend(ids)
            token_types.extend(type_labels)

    return all_ids, token_types

def tokenize_list_with_biprior (
        knowledge_list: List[str],
        front_priors: List[str],
        back_priors: List[str],
        model: str):
    
    tuple_list = list(zip(front_priors, knowledge_list, back_priors))

    random.shuffle(tuple_list)
    tokenizer = get_tokenizer(model)
    all_ids = []
    token_types = []

    for data in tqdm(tuple_list):
        if data:
            front_prior = data[0]
            knowledge = data[1]
            back_prior = data[2]
            
            front_encoded = tokenizer.encode(front_prior, add_special_tokens=True)
            knowledge_encoded = tokenizer.encode(knowledge, add_special_tokens=False)
            back_encoded = tokenizer.encode(back_prior, add_special_tokens=False)

            ids = front_encoded + knowledge_encoded + back_encoded
            ids.append(tokenizer.eos_token_id)

            front_length = len(front_encoded)
            knowledge_length = len(knowledge_encoded)
            back_length = len(back_encoded)

            type_labels = (
                [PRIOR_TOKEN_TYPE] * front_length +
                [KNOWLEDGE_ARTICLE_TOKEN_TYPE] * knowledge_length +
                [PRIOR_TOKEN_TYPE] * (back_length + 1)  # +1 for EOS
                )
            
            assert len(ids) == len(type_labels)

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

    # TODO: Verify the path is correct
    df = pd.read_pickle(os.path.join(ROOT, f'dataset/priorlearning/alpha/prior_alpha_dataset_{model_name}.pickle'))
    knowledge_article, front_priors, back_priors = df['article'], df['front_prior'], df['back_prior']
    nontarget_content = pd.read_pickle(os.path.join(ROOT, 'dataset/nontarget_article/alpha/nontarget_article_table.pickle'))['content']

    dir = os.path.join(ROOT, f'training/frontprior_data/{model_nickname}')
    if not os.path.isdir(dir):
        os.mkdir(dir)

    #TODO: Verify the function is correct
    all_ids, token_type = tokenize_list_with_frontprior(knowledge_article, front_priors, back_priors, model)
    write_to_memmap_single(all_ids, dir, f'knowledge_article.bin')
    write_to_memmap_single(token_type, dir, f'knowledge_article_token_type.bin')
   
   
    all_ids, token_type = tokenize_list(nontarget_content, model)
    write_to_memmap_single(all_ids, dir, f'nontarget_content.bin')
    write_to_memmap_single(token_type, dir, f'nontarget_content_token_type.bin')

if __name__ == '__main__':
    model_names = ["llama3.1-8B-instruct", "mistral-7b-instruct-v0.3"]
    for model_name in model_names:
        print(f'Tokenizing for {model_name}')
        tokenize_knowledge(model_name=model_name)