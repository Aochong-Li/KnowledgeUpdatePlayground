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

current_date = datetime.datetime.now(
        pytz.timezone("America/New_York")
    ).strftime("%B %d, %Y")

sys.path.insert(0 , '/home/al2644/research/')
sys.path.insert(1, '/home/al2644/research/codebase')

SERPAPI_KEY = 'LAeuUqFiboyp9Tc4R61micUqNLGHfryVaUjlVSaverZyFaVl28rX8odECE5T'
output_dir = '/share/goyal/lio/knowledge_delta/dataset/nontarget_article'

ENTITY_TABLE = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/entity/entity_pool.pickle')
ENTITY_DICT = ENTITY_TABLE[['entity', 'entity_id']].set_index('entity')['entity_id'].to_dict()

def call_live_engine(query:str, search_type:str = 'news', num_results: int = 100, page: int = 1):
    url = "https://api.serphouse.com/serp/live"
    params = {
        "api_token": SERPAPI_KEY,
        "q": query,
        "num_result": num_results,   
        "domain": "google.com",  
        "lang": "en",  
        "device": "desktop",
        "serp_type": search_type,    # web or news
        "loc": "New York,United States",
        "page": page
    }
    response = requests.get(url, params=params)
    
    return response

def call_schedule_engine (queries: list, search_type: str = 'news', num_results: int = 100, page: int = 1):
    assert len(queries) <= 100, "SERPHouse API cannot handle Too Many Requests"

    url = 'https://api.serphouse.com/serp/schedule'
    headers = {
    'Authorization': f'Bearer {SERPAPI_KEY}',
    'Content-Type': 'application/json'
    }
    
    task_template = {
        "domain": "google.com",
        "lang": "en",
        "device": "desktop",
        "serp_type": search_type,
        "loc": "New York,United States",
        "postback_url": "https://your-webhook-url.com",
        "page": page,
        "num_result": num_results
    }

    task_data = {"data": []}
    for query in queries:
        task = task_template.copy()
        task["q"] = query

        task_data['data'].append(task)

    response = requests.post(url, headers=headers, data=json.dumps(task_data))
    if response.status_code == 200:
        response_data = response.json()
        task_ids = response_data['results']
        return task_ids
    else:
        print(f"Failed to schedule task for {queries}")
        return None

def get_article_content(reference_url):
    try:
        article = Article(reference_url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""
    
def retrieve_article (metadata: dict):
    url = metadata['url']
    content = get_article_content(url)
    if content != "":
        metadata['content'] = content
        return metadata

def download_news_schedule_parallel (entity_list: list, fname: str, chunk_size: int = 100, pages: int = 1):
    ###TODO: This is a hack
    # update_table = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/update/update_table.pickle')
    # entity_table = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/entity/entity_table.pickle')
    # df = entity_table.merge(update_table, on = ['entity_id'])

    # entity_list = list(df['entity'])
    ###

    import pdb;pdb.set_trace()
    chunks = [entity_list[i:i+chunk_size] for i in range(0, len(entity_list), chunk_size)]

    results = []
    for page in range(1, pages + 1): # don't change this if you want to scrape from the 1st page
        print(f'Scrapping News on Page {page}')
        for i, chunk_data in enumerate(chunks):
            print(f'Processing {i}th chunk of 100 queries')
            task_ids = call_schedule_engine(queries=chunk_data, search_type='news', num_results=100, page=page)
            results.extend(task_ids)

            with open(os.path.join(output_dir, fname), 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL) 

            time.sleep(120)

def retrieve_news_parallel (all_tasks: list, output_subdir = 'google_news_plus', num_processes: int = 20):
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

        return metadata
    
    def retrieve_articles_from_metadata (metadata: list):
        results = []
        for m in metadata:
            result = retrieve_article(m)
            if result:
                results.append(result)
        
        return results
    
    def process_tasks (tasks, queue, progress_bar_position):
        failed_tasks = []
        for task in tqdm(tasks, desc=f"Process-{progress_bar_position}", position=progress_bar_position):
            entity = task['q']
            entity_id = ENTITY_DICT.get(entity, entity)
            task_id = task['id']

            try:
                filepath = os.path.join(output_dir, f'{output_subdir}/{entity_id}.pickle')
                if os.path.isfile(filepath):
                    result = pd.read_pickle(filepath)
                else:
                    result = []
                metadata = retrieve_metadata(task_id=task_id)
                if metadata == []:
                    failed_tasks.append(task)
                    continue
                
                articles = retrieve_articles_from_metadata(metadata)
                result.extend(articles)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                print(f"Error while processing task {task}: {e}")
                failed_tasks.append(task)
        
        print(f'Success: {len(tasks) - len(failed_tasks)} / {len(tasks)}')
        queue.put(failed_tasks)

    chunk_size = max(1, len(all_tasks) // num_processes)
    chunks = [all_tasks[i:i+chunk_size] for i in range(0, len(all_tasks), chunk_size)]
    
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    processes = []

    for i, chunk_data in enumerate(chunks):
        p = multiprocessing.Process(target=process_tasks, args = (chunk_data, queue, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    failed_tasks = []
    while not queue.empty():
        failed_tasks.extend(queue.get())
    
    with open(os.path.join(output_dir, 'serp_tasks/alpha_additional_failed_tasks_page1.pickle'), 'wb') as f:
        pickle.dump(failed_tasks, f, protocol=pickle.HIGHEST_PROTOCOL)

def convert_to_pandas (root: str, output_fname: str):
    results = []

    def remove_duplicates(articles: list, entity_id: int):
        df = pd.DataFrame(articles)
        df = df.dropna(subset=['content']).drop_duplicates(subset=['url'], keep='first')
        df['entity_id'] = entity_id

        # Filtering
        message = "your browser supports JavaScript and cookies"
        df = df[~df['content'].str.contains(message)]

        return df[['entity_id', 'url', 'time', 'content']]
    
    for fname in tqdm(os.listdir(root)):
        entity_id = int(fname.replace('.pickle', ''))
        articles = pd.read_pickle(os.path.join(root, fname))
        
        if len(articles) == 0:
            continue

        df = remove_duplicates(articles=articles, entity_id=entity_id)
        results.append(df)

    results = pd.concat(results).reset_index(drop=True)

    results.to_pickle(os.path.join(output_dir, output_fname))

if __name__ == "__main__":
    import pdb; pdb.set_trace()

    '''Scrape URLs'''
    # fact_table = pd.read_pickle("/share/goyal/lio/knowledge_delta/dataset/update/alpha/fact_table.pickle")
    # current_nontarget = pd.read_pickle("/share/goyal/lio/knowledge_delta/dataset/nontarget_article/nontarget_article_pool.pickle")
    # entity_pool = pd.read_pickle("/share/goyal/lio/knowledge_delta/dataset/entity/entity_pool.pickle")

    # remain_entity_ids = set(fact_table['entity_id']).difference(current_nontarget['entity_id'])
    # entity_list = list(entity_pool[entity_pool['entity_id'].isin(remain_entity_ids)]['entity'])
    # fname = "serp_tasks/alpha_additional_tasks_page1.pickle"

    # download_news_schedule_parallel(entity_list, fname)

    '''Retrieve Articles'''
    # tasks = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/nontarget_article/serp_tasks/alpha_additional_tasks_page1.pickle')
    # output_subdir = "alpha/alpha_additional_google_news"
    # retrieve_news_parallel(tasks, output_subdir, 20)

    '''Concatenate Articles in DataFrame'''
    root = '/share/goyal/lio/knowledge_delta/dataset/nontarget_article/google_news'
    convert_to_pandas(root=root, output_fname="beta_tables/nontarget_article_pool.pickle")