import os
import sys
import pandas as pd
from tqdm import tqdm
import json
import requests

import multiprocessing
from newspaper import Article
import datetime
import pytz
import pickle
import time
import pdb
from collections import defaultdict

SERPAPI_KEY = 'LAeuUqFiboyp9Tc4R61micUqNLGHfryVaUjlVSaverZyFaVl28rX8odECE5T'
output_dir = '/share/goyal/lio/knowledge_delta/dataset/nontarget_article'

ENTITY_TABLE = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/entity/entity_pool.pickle')
ENTITY_DICT = ENTITY_TABLE[['entity', 'entity_id']].set_index('entity')['entity_id'].to_dict()

def clean_results (all_tasks: list, output_subdir = 'google_news'):
    '''
    clean_results seems to be redundant given convert_to_pandas in scrape_news.py
    '''
    def remove_duplicates(articles: list):
        df = pd.DataFrame(articles)
        df = df.dropna(subset=['content']).drop_duplicates(subset=['url'], keep='first')

        return df[['url', 'content']]

    def retrieve_metadata (task_id: int):
        url = f'https://api.serphouse.com/serp/get?id={task_id}'
        headers = {
            'Authorization': f'Bearer {SERPAPI_KEY}',
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers)
        if response.json()['msg'] == 'Keyword is not processed yet.':
            print('keyword not processed yet')
            return []
            
        news_objs = response.json()['results']['results']['news']
        metadata = [{'url': news['url'], 'time': news['time']} for news in news_objs if 'url' in news and 'time' in news]
        metadata_df = pd.DataFrame(metadata)

        return metadata_df
    
    for task in tqdm(all_tasks):
        entity = task['q']
        entity_id = ENTITY_DICT.get(entity, entity)
        task_id = task['id']

        try:
            filepath = os.path.join(output_dir, f'{output_subdir}/{entity_id}.pickle')
            if os.path.isfile(filepath):
                articles = pd.read_pickle(filepath)
            else:
                continue

            # deduplicate 
            dedup_article_df = remove_duplicates(articles)
            metadata_df = retrieve_metadata(task_id=task_id)
            
            result_df = dedup_article_df.merge(metadata_df, on = ['url'])
            result = result_df.to_dict(orient = 'records')
            
            with open(filepath, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            index = all_tasks(task)
            raise Exception(f'Failed at {index}')
        
def filter_nontarget_articles (filepath, min_len: int = 1000, min_articles: int = 30, max_articles: int = 50):
    '''
    filter out short articles (< 1000 in length)
    fitler out entities without enough articles
    '''
    df = pd.read_pickle(filepath)
    df['len'] = df['content'].str.len()
    df = df[df['len'] > min_len].drop(columns = ['len'])
    entities = df.groupby('entity_id').count()['url']
    entities = entities[entities > min_articles].reset_index().drop(columns = ['url'])
    import pdb; pdb.set_trace()
    df = df[df['entity_id'].isin(set(entities['entity_id']))]
    df = df.groupby('entity_id').apply(
            lambda x: x.sample(n=min(len(x), max_articles), random_state=42)
        ).reset_index(drop = True)
    df.to_pickle('/share/goyal/lio/knowledge_delta/dataset/nontarget_article/alpha/nontarget_article_table.pickle')

if __name__=='__main__':
    filepath = '/share/goyal/lio/knowledge_delta/dataset/nontarget_article/alpha/nontarget_article_pool.pickle'
    filter_nontarget_articles(filepath)
    # tasks = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/nontarget_article/tasks_page1.pickle')
    # clean_results(tasks)



