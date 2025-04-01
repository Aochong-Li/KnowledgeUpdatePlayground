import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.gpt_engine import GPT_Engine
from codebase.knowledge_update.serpapi import scrape_google
from prompts import *
from constants import TOGETHERAI_MODEL_LIST

output_dir = '/share/goyal/lio/knowledge_delta/dataset'

def generate_fact_candidates (cache_filepath: str,
                    model = 'gpt-4o',
                    temperature = 1.0,
                    max_tokens = 1024,
                    ):
    prompt_template = STEP1_GENERATE_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP1_SYSTEM_PROMPT

    input_df = pd.read_pickle(os.path.join(output_dir, 'entity/entity_table.pickle'))
    template_map = {'category': 'category', 'entity': 'entity'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_facts',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      )

def judge_fact_candidates (input_filepath: str,
                 cache_filepath: str,
                 model = 'gpt-4o-mini',
                 temperature = 1.0,
                 max_tokens = 1024,
                 ):
    prompt_template = STEP1_JUDGE_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP1_SYSTEM_PROMPT

    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_table.pickle'))
    fact_df  = pd.read_pickle(input_filepath)
    input_df = entity_table[['entity', 'entity_id']].merge(fact_df, on = ['entity_id'])
    input_df = input_df.explode('fact', ignore_index = True)

    template_map = {'entity': 'entity', 'fact': 'fact'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='judge_facts',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      )

def filter_fact (input_filepath: str,
                 cache_filepath: str,
                 model: str,
                 temperature: float = 1.0,
                 max_tokens: int = 128,
                 n: int = 10
                 ):
    prompt_template = STEP1_FILTER_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP1_SYSTEM_PROMPT

    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_pool.pickle'))
    fact_df  = pd.read_pickle(input_filepath)
    input_df = entity_table[['entity_id', 'entity', 'category']].merge(fact_df, on = ['entity_id'])

    template_map = {'entity': 'entity', 'category': 'category', 'fact': 'fact'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='filter_facts',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      n = n)

''' Beta Utils for this stage'''
def parse_fact_candidates ():
    fact_df = pd.read_pickle(os.path.join(output_dir, 'update/candidate/generate_candidates.pickle'))[['response']]
    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_table.pickle'))
    fact_df = entity_table[['entity_id']].merge(fact_df, left_index = True, right_index = True)

    def parse_list (response: str):
        clean_response = response.replace('```', '').replace('python', '')
        try:
            clean_response = clean_response.split('=')[1].strip()
            return eval(clean_response)
        except:
            return None
    fact_df['fact'] = fact_df['response'].apply(parse_list)
    fact_df = fact_df[fact_df['fact'].notnull()]
    fact_df.to_pickle(os.path.join(output_dir, 'update/candidate/generate_candidates.pickle'))

def parse_fact_judgement (save_filename = 'fact_candidates.pickle'):
    def parse_response(response: str):
        try:
            lines = [line for line in response.lower().split('\n') if 'label' in line and ('good' in line or 'bad' in line)]
            label = lines[-1]
        except:
            return None
        
        if 'good' in label and 'bad' in label:
            return None
        elif 'good' in label:
            return True
        else:
            return False
        
    fact_df = pd.read_pickle(os.path.join(output_dir, 'update/candidate/generate_candidates.pickle'))
    judge_df = pd.read_pickle(os.path.join(output_dir, 'update/candidate/judge_candidates.pickle'))
    judge_df['judgement'] = judge_df['response'].apply(parse_response)

    result_df = fact_df[['entity_id', 'fact']].explode('fact', ignore_index = True).merge(judge_df[['judgement']], left_index = True, right_index = True)
    result_df['_id'] = result_df.groupby('entity_id').cumcount()
    result_df['fact_id'] = result_df['entity_id'].astype(str) + '_' + result_df['_id'].astype(str)
    result_df.drop(columns = ['_id'], inplace = True)

    result_df.to_pickle(os.path.join(output_dir, f'update/candidate/{save_filename}'))

def parse_fact_filter (dir: str, merge: bool):
    def convert (responses: list):
        result = []

        if responses is None or len(responses) == 0:
            return None
        
        for response in responses:
            if 'true' in response.lower() and 'false' not in response.lower():
                result.append(True)
            else:
                result.append(False)
                
        ratio = sum(result)/ len(result)
        
        return True if ratio > 0.5 else False
    
    filters = []

    for fname in os.listdir(dir):
        col = fname.replace('.pickle', '')
        df = pd.read_pickle(os.path.join(dir, fname))
        df[col] = df['response'].apply(convert)
        filters.append(df[[col]])
    
    filters = pd.concat(filters, axis = 1)
    filters['All_True'] = filters.all(axis = 1)
    filters['All_False'] = filters.any(axis = 1)
    filters.to_pickle(os.path.join(dir, 'summary.pickle'))

    if merge:
        update_table = pd.read_pickle(os.path.join(output_dir, 'update/update_table.pickle')).reset_index(drop = True)
        assert len(update_table) == len(filters)

        update_table = update_table.merge(filters[['All_True']], left_index = True, right_index = True)
        update_table = update_table[update_table['All_True']].drop(columns = ['All_True'])

        update_table.to_pickle(os.path.join(output_dir, 'update/update_table.pickle'))

def sample_fact (input_filename:str = 'fact_candidates.pickle', save_filename:str = 'update_table.pickle', random_state: int = 42):
    candidate_df = pd.read_pickle(os.path.join(output_dir, f'update/candidate/{input_filename}')).dropna(subset = ['judgement'])
    sample_df = candidate_df[candidate_df['judgement']].groupby('entity_id').sample(n = 1, random_state = random_state)
    sample_df = sample_df[['entity_id', 'fact_id', 'fact']].reset_index(drop = True)

    sample_df.to_pickle(os.path.join(output_dir, f'update/{save_filename}'))

'''manual filtering'''
def manual_filter (output_filepath: str = os.path.join(output_dir, "update/alpha/candidate/fact_candidates.pickle")):
    remove_list = {
        "people": ["resides in", "resident of", "active on social media", "media platforms"],
        "companies": ["headquarter", "CEO", "listed", "stock exchange", "publicly traded"],
        "sports": ["compete in", "owned by", "ownership", "team color", "stadium"],
        "media series": ["produced by", "published by", "developed by", "available for"]
    }
    
    restrict_list = {
        "buildings & landmarks": ["located"],
        "institutions": ["headquarter"],
    }
    
    bias_remove_list = ["actively ", "frequently ", "currently ", "primarily ", "available for "]

    entity_pool = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/entity/entity_pool.pickle')
    fact_candidate_df = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/update/candidate/fact_candidates.pickle')
    fact_candidate_df = fact_candidate_df[fact_candidate_df['judgement'] == True]
    df = entity_pool.merge(fact_candidate_df)
    
    output_df = []
    for category in df['category'].unique():
        category_df = df[df['category'] == category]
        
        if category in remove_list:
            keywords = '|'.join(remove_list[category])
            category_df = category_df[~category_df['fact'].str.contains(keywords, case=False)]

        if category in restrict_list:
            keywords = '|'.join(restrict_list[category])
            common_df = category_df[category_df['fact'].str.contains(keywords)].sample(n = 5)
            category_df = pd.concat([common_df, category_df[~category_df['fact'].str.contains(keywords)]], ignore_index=True)
        
        output_df.append(category_df)

    output_df = pd.concat(output_df, ignore_index=True)
    output_df['fact'] = output_df['fact'].replace(bias_remove_list, '', regex=True)

    output_df[['entity_id', 'fact']].to_pickle(output_filepath)


if __name__=='__main__':
    parse_fact_filter("/share/goyal/lio/knowledge_delta/dataset/update/alpha/filter_update", False)


    # for _, model in TOGETHERAI_MODEL_LIST.items():
    #     print(f'Assesssing {model}')
    #     fact_filepath = '/share/goyal/lio/knowledge_delta/dataset/update/update_table.pickle'
    #     cache_filepath = f'/share/goyal/lio/knowledge_delta/dataset/update/filter/{model.split('/')[1]}.pickle'
        
    #     engine = filter_fact(fact_filepath, cache_filepath, model)
    #     engine._run_model(num_processes=10)

    # sample_fact(random_state=42)
    # input_filepath = '/share/goyal/lio/knowledge_delta/dataset/update/candidate/generate_candidates.pickle'
    # cache_filepath = '/share/goyal/lio/knowledge_delta/dataset/update/candidate/judge_candidates.pickle'
    # engine = judge_fact_candidates(input_filepath=input_filepath, cache_filepath=cache_filepath)
    # engine._run_model(num_processes=30)
                
# def verify_facts (df, entity_col = 'title', wikitext_col = 'text', fact_col = 'step1_output',
#                   model = 'gpt-4o-mini', temperature = 0.5, max_tokens = 1024, batch_size = 10,
#                   input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
    
#     input_prompt_template = STEP1V_INPUT_PROMPT_TEMPLATE
#     system_prompt = SYSTEM_PROMPT

#     template_properties = {
#         'entity': entity_col,
#         'wikitext': wikitext_col,
#         'facts': fact_col
#                            }
    
#     run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
#             template_properties = template_properties, model = model, temperature = temperature,
#             max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
#             batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)

# def check_facts (df, entity_col = 'title', fact_col = 'step1v_output',model = 'gpt-4o-mini',
    #              temperature = 0.5, max_tokens = 512, batch_size = 30,
    #              input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
    
    # input_prompt_template = STEP1C_INPUT_PROMPT_TEMPLATE
    # system_prompt = SYSTEM_PROMPT

    # template_properties = {
    #     'entity': entity_col,
    #     'fact': fact_col
    #                        }
    
    # run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
    #         template_properties = template_properties, model = model, temperature = temperature,
    #         max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
    #         batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)

# def google_search (input_df, col = 'step1_output', num_workers = 20, cache_filepath = ''):
#     queries = list(input_df[col])
#     results = []
#     failed_results = []

#     with Pool(num_workers) as pool:
#         for result in tqdm(pool.imap_unordered(scrape_google.organic_search_engine, queries), total=len(queries)):
#             if isinstance(result, str):
#                 failed_results.append(result)
#             else:
#                 results.append(result)
    
#     with open(cache_filepath, 'wb') as f:
#         results = {key: value for d in results for key, value in d.items()}
#         pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#     return failed_results

# def format_evidence (input_df, search_result_filepath):
#     evidence_list = []
#     search_results = pd.read_pickle(search_result_filepath)

#     for i in range(len(input_df)):
#         fact, wiki = input_df['step1_output'].iloc[i], input_df['text'].iloc[i]
#         evidence = f'source: Wikipedia\ncontent: {wiki}'

#         sources = search_results[fact]
#         if isinstance(sources, str):
#             evidence_list.append(evidence)
#             continue
#         else:
#             for d in sources:
#                 evidence += f'\n\nsource: {d['site_title']}\ncontent: {d['snippet']}'
        
#         evidence_list.append(evidence)

#     input_df['evidence'] = evidence_list
#     return input_df