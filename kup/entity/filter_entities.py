import os
import pandas as pd
import numpy as np
import sys

import multiprocessing
import time
import random
from typing import List

from serp.scrape_google import *
import constants

import wikipediaapi

output_dir = "YOUR_OUTPUT_DIR_PATH"

'''
Step I: Scrape Wikipedia Page for Each Entity
'''
def wiki_entity_candidates (
        filepath: str,
        num_chunks: int = 10
        ):
    df = pd.read_pickle(filepath)
    chunk_size, i = max(1, len(df) // num_chunks), 1479
    
    while i < len(df):
        chunk_df = df.iloc[i:i+chunk_size]
        print(f'Processing {i} to {i + chunk_size}')

        get_entity_wiki(entities=list(chunk_df['entity']), filename=f'chunk{i}.pickle')

        i += chunk_size
        time.sleep(10)

def concat_candidate_chunks ():
    df = []
    for fname in os.listdir(output_dir):
        if 'chunk' not in fname:
            continue
        df.append(pd.read_pickle(os.path.join(output_dir, fname)))
    df = pd.concat(df).reset_index(drop=True)

    df = df[(df['wiki_page'] != '') & (df['wiki_metadata'].notnull())]

    # Hack to remove some wiki pages
    def remove_category_page (metadata: dict):
        if 'https://en.wikipedia.org/wiki/Category:' in metadata['link']:
            return False
        else:
            return True
    df = df[df['wiki_metadata'].apply(remove_category_page)].reset_index(drop = True)


    df.to_pickle(os.path.join(output_dir.replace('/wiki', ''), 'candidates.pickle'))

def get_entity_wiki (entities: List[str], num_processes: int = 10, filename: str = 'candidates.pickle'):
    # pre-processing: form queries from entities
    queries = [f'Wiki {entity}' for entity in entities]

    chunk_size = max(1, len(queries) // num_processes)
    chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    processes = []

    def _call_scrape_wiki_page (chunk, queue, progress_bar_position):
        results = []

        for query in tqdm(chunk, desc=f"Process-{progress_bar_position}", position=progress_bar_position):
            time.sleep(0.5)
            result = scrape_wiki_page(query, num_results = 100, max_retries=3)
            results.append(result)

        queue.put(results)

    for i, chunk_data in enumerate(chunks):
        p = multiprocessing.Process(target=_call_scrape_wiki_page, args=(chunk_data, queue, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    all_results = []
    while not queue.empty():
        all_results.extend(queue.get())

    df = pd.DataFrame(all_results, columns = ['entity', 'wiki_metadata', 'wiki_page']).reset_index(drop=True)
    # process entity column
    df['entity'] = df['entity'].apply(lambda x: x.replace('Wiki ', '') if isinstance(x, str) else x)
    df.to_pickle(os.path.join(output_dir, filename))

def scrape_wiki_page (query: str, num_results: int = 20, max_retries: int = 3):    
    def get_wiki_metadata(query=query, num_results=num_results):
        response = call_live_engine(query=query, num_results=num_results)
        try:
            organic_results = response.json()['results']['results']['organic']
            wiki_result = [result for result in organic_results 
                            if 'https://en.wikipedia.org' in result['displayed_link'] 
                            and 'https://en.wikipedia.org/wiki/Category:' not in result['displayed_link']
                            and 'link' in result.keys()
                            ]
            return wiki_result[0]
        except:
            return None
        
    def get_clean_wiki_page(url: str):
            wiki_wiki = wikipediaapi.Wikipedia(user_agent = "wikipage/1.0 (al4143@cornell.edu)", language = 'en')
            page_title = url.split('/')[-1]
            page = wiki_wiki.page(page_title)

            if page.exists():
                return page.text
            else:
                import requests
                from bs4 import BeautifulSoup
                response = requests.get(url)
                try:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    wiki_page = soup.get_text()
                    parsing_keyword='From Wikipedia, the free encyclopedia'

                    if parsing_keyword:
                        clean_wiki_page = wiki_page.split(parsing_keyword)[1].split('References')[0]
                        return clean_wiki_page
                    else:
                        return wiki_page
                except:
                    return None
    
    def scrape_attempt():
        wiki_metadata = get_wiki_metadata()
        if wiki_metadata is None:
            return (query, None, None)
        else:
            wiki_page = get_clean_wiki_page(wiki_metadata['link'])
            return (query, wiki_metadata, wiki_page)

    # SERPHOUSE API    
    attempt = 0
    while attempt < max_retries:
        try:
            results = scrape_attempt()

            if results[1] is not None and results[2] is not None:
                return results
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
        
        attempt += 1
        time.sleep(2 ** attempt + random.random())
    
    print(f"All retries failed for {query} with SERPHouseAPI. Now trying googlesearch")

    return (query, None, None)

'''
STEP II: Generate Wikipedia by Model
'''
def recreate_wiki_page (model:str,
                       temperature: float = 0.7,
                       max_tokens:int = 2048):
    from evaluation.llm_evaluator import Evaluator

    prompt_template = '''You are a knowledgeable assistant with memories of all Wikipedia articles.

Task: Write a detailed, objective, and comprehensive Wikipedia-style article on {entity} using all the factual details you can recall (including dates, numbers, names, events, etc.).

Guidelines:
    1. Present the information in a neutral, encyclopedic tone, similar to Wikipedia.
    2. Include relevant subheadings or sections (e.g., Background, History, Key Events, Impact, etc.) as needed.
    '''

    template_map = {'entity': 'entity'}
    input_df = pd.read_pickle(os.path.join(output_dir, 'candidates.pickle'))
    cache_filepath = os.path.join(output_dir, f'{model}.pickle')

    engine = Evaluator(
            df=input_df, 
            prompt_template=prompt_template,
            template_map=template_map,
            model_name=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    result = engine._chat()

    result.to_pickle(cache_filepath)

def compute_rouge2 (model: str):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    
    filepath = os.path.join(output_dir, f'{model}.pickle')
    assert os.path.isfile(filepath), f'{filepath} does not exist'

    entity_df = pd.read_pickle(os.path.join(output_dir, 'candidates.pickle'))
    model_response_df = pd.read_pickle(filepath)

    assert len(entity_df) == len(model_response_df), f'{model} does not have {len(entity_df)} responses'
    print(f'Computing ROUGE2 score for {model}')

    # Compute ROUGE-2 precision and F-measure
    precision_scores = []
    fmeasure_scores = []

    for i in tqdm(range(len(entity_df))):
        score = scorer.score(entity_df['wiki_page'].iloc[i], model_response_df['response'].iloc[i])['rouge2']
        precision_scores.append(score.precision)
        fmeasure_scores.append(score.fmeasure)
    
    # Add the new columns to the entity dataframe
    entity_df[f'{model}_rouge2_precision'] = precision_scores
    entity_df[f'{model}_rouge2_fmeasure'] = fmeasure_scores

    entity_df.to_pickle(os.path.join(output_dir, 'candidates.pickle'))

def filter_candidates (
        threshold: float = 0.05,
        len_qtile: float = 0.05,
        metric: str = 'fmeasure',
        fname: str = 'entity_pool.pickle'
        ):
    '''
    Guideline: For each model, we filter
        1. unable to respond message
        2. filter short response, which will make metric a more robust measurement
        3. the metric score below the threshold
    '''
    models = [model for model in constants.MODEL_LIST if 'instruct' in model]

    def filter_model_response (model: str):
        failure_messages = [
            # llama3.2-3B-instruct
            "can't provide information",
            "unable to provide information",
            "don't have information",
            "don't have any information",
            
            # llama3.1-8B-instruct
            "couldn't find any information",
            'no information available',
            'unable to verify',
            'unable to provide information',
            # mistral-7b-instruct-v0.3 no explicit refusal message

            # gemma2-9b-instruct        
            'i do not have access to',
            "i can't access",
            'i apologize, but i',
            'i am sorry, but i',
            'please provide me with',
            'please note: '
        ]
            
        model_response = pd.read_pickle(os.path.join(output_dir, f'{model}.pickle'))
        model_response['response_len'] = model_response['response'].str.len()

        clean_response = model_response[model_response['response'].apply(lambda x: not any(msg in x.lower() for msg in failure_messages))]
        len_threshold = clean_response['response_len'].quantile(len_qtile)
        clean_response = clean_response[clean_response['response_len'] > len_threshold]

        return set(clean_response.index)
    
    clean_response_index = [filter_model_response(model) for model in models]
    clean_response_index = list(set.intersection(*clean_response_index))

    entity_df = pd.read_pickle(os.path.join(output_dir, 'candidates.pickle'))
    clean_entity_df = entity_df.loc[clean_response_index]

    metric_col = [col for col in clean_entity_df.columns if metric in col]
    clean_entity_df = clean_entity_df[(clean_entity_df[metric_col] > threshold).all(axis = 1)]

    category_df = pd.read_pickle(os.path.join(output_dir, 'candidate/dedup_candidates.pickle'))
    result = category_df[category_df['entity'].isin(clean_entity_df['entity'])].reset_index(drop = True)
    result = result.reset_index().rename(columns = {'index': 'entity_id'})

    result.to_pickle(os.path.join(output_dir, fname))

'''
STEP III (Optional): Sample Entities + Add Page Views
'''    
def sample_candidates (filepath: str, n: int = 100, seed: int = 42):
    import pdb; pdb.set_trace()
    df = pd.read_pickle(os.path.join(output_dir, 'entity_table.pickle'))
    nontarget_article_df = pd.read_pickle(os.path.join('/share/goyal/lio/knowledge_delta/dataset/nontarget_article/nontarget_article_table.pickle'))[['entity_id']].drop_duplicates()
    df = df.merge(nontarget_article_df)

    sample_df = df.groupby('category')[['entity_id', 'entity']].apply(lambda x: x.sample(n = n, random_state = seed)).sort_values('entity_id').reset_index().drop(columns = ['level_1'])

    sample_df.to_pickle(filepath)

def add_wiki_pageview(filepath: str, start_date: str = '20210101', end_date: str = '20231231'):
    import pageviewapi
    entity_table = pd.read_pickle(filepath)
    wiki_table = pd.read_pickle(os.path.join(output_dir, 'candidates.pickle'))[['entity', 'wiki_metadata']]

    entity_table = entity_table.merge(wiki_table, on =['entity'])

    def _get_monthly_view (metadata):
        title = metadata['title']
        if "- Wikipedia" in title:
            title = title.replace('- Wikipedia', '').strip()

        if "&#39;" in title:
            title = title.replace('&#39;', "'")

        try:
            views = pageviewapi.per_article(
                'en.wikipedia',      # The language and project (e.g., 'en.wikipedia')
                title,               # The title of the Wikipedia page
                start_date,          # Start date in 'YYYYMMDD' format
                end_date,            # End date in 'YYYYMMDD' format
                access='all-access', # 'all-access', 'desktop', 'mobile-app', or 'mobile-web'
                agent='all-agents',  # 'all-agents', 'user', 'spider', or 'bot'
                granularity='monthly'  # 'daily' or 'monthly'
            )

            monthly_view  = np.mean([item['views'] for item in views['items']])

            return monthly_view
        except:
            return None
        
    entity_table['monthly_pageview'] = entity_table['wiki_metadata'].apply(_get_monthly_view)

    entity_table = entity_table.drop(columns = ['wiki_metadata'])
    entity_table.to_pickle(filepath)

if __name__ == '__main__':
    """
    Call Each Step to Filter Entity Candidates
    """
