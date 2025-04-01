import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')
sys.path.insert(1 , '/home/al2644/research/codebase/knowledge_update/dataset_creation')

from codebase.knowledge_update.gpt_engine import GPT_Engine
from prompts import *

output_dir = '/share/goyal/lio/knowledge_delta/dataset'

def generate_guidelines (cache_filepath: str,
                         model = 'gpt-4o',
                         temperature = 1.0,
                         max_tokens = 2048
                         ):
    prompt_template = STEP3_GENERATE_GUIDELINE_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP3_SYSTEM_PROMPT
    # input_df = pd.read_pickle('/share/goyal/lio/knowledge_delta/dataset/article/alpha/candidate/remain_job.pickle')
    import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    entity_pool = pd.read_pickle(os.path.join(output_dir, 'entity/entity_pool.pickle'))
    input_df = pd.read_pickle(os.path.join(output_dir, 'update/alpha/update_table.pickle'))

    input_df = entity_pool.merge(input_df, on = ['entity_id']).reset_index(drop = True)

    template_map = {'entity': 'entity', 'fact': 'fact', 'update': 'update'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_guidelines_cont',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode='chat_completions'
                      )

def generate_base_articles (cache_filepath: str,
                      model = 'gpt-4o',
                      temperature = 1.0,
                      max_tokens = 2048
                      ):
    import pdb; pdb.set_trace()
    prompt_template = STEP3_GENERATE_ARTICLE_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP3_SYSTEM_PROMPT

    entity_pool = pd.read_pickle(os.path.join(output_dir, 'entity/entity_pool.pickle'))
    update_table = pd.read_pickle(os.path.join(output_dir, 'update/alpha/update_table.pickle'))

    input_df = entity_pool.merge(update_table, on = ['entity_id'])

    template_map = {'entity': 'entity', 'update': 'update', 'audience': 'audience'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_base_articles',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )

    # excerpt = pd.read_pickle(os.path.join(output_dir, 'article/excerpt.pickle'))

    # def sample_excerpt (entity_id: str):
    #     if entity_id in excerpt['entity_id'].unique():
    #         return excerpt[excerpt['entity_id'] == entity_id]['excerpt'].sample(n = 1).values[0]
    #     else:
    #         return excerpt['excerpt'].sample(n = 1).values[0]

    # input_df['excerpt'] = input_df['entity_id'].apply(sample_excerpt)

def build_excerpt_df (filepath: str, min_len: int = 2000):
    import pdb; pdb.set_trace()
    df = pd.read_pickle(filepath)
    df['len'] = df['content'].str.len()
    df = df[df['len'] > min_len]
    df['excerpt'] = df['content'].apply(lambda x: x[:min_len] + ' ...')

    df.drop(columns = ['len']).to_pickle(os.path.join(output_dir, 'article/alpha/excerpt.pickle'))

def generate_refined_articles (cache_filepath: str,
                      model = 'gpt-4o',
                      temperature = 1.0,
                      max_tokens = 2048
                      ):
    import pdb; pdb.set_trace()
    prompt_template = STEP3_REFINE_ARTICLE_INPUT_PROMPT_TEMPLATE
    system_prompt = STEP3_SYSTEM_PROMPT

    entity_pool = pd.read_pickle(os.path.join(output_dir, 'entity/entity_pool.pickle'))
    update_table = pd.read_pickle(os.path.join(output_dir, 'update/alpha/update_table.pickle'))
    audience = pd.read_pickle(os.path.join(output_dir, 'article/alpha/audience_group.pickle'))
    base_article = pd.read_pickle(os.path.join(output_dir, 'article/alpha/base_article_table.pickle'))

    input_df = entity_pool.merge(update_table, on='entity_id').merge(base_article, on='entity_id').merge(audience, on='entity_id')

    # INTRODUCE EXERCEPT

    def assign_excerpt (article_group):
        entity_id = article_group.name
        num_articles = len(article_group)

        excerpts = excerpt_df.loc[excerpt_df['entity_id'] == entity_id, 'excerpt'].drop_duplicates()

        assert len(excerpts) >= num_articles, f"Not Enough Excerpts for {entity_id}"

        article_group['excerpt'] = list(excerpts.sample(n = num_articles, random_state = 42))

        return article_group

    excerpt_df = pd.read_pickle(os.path.join(output_dir, 'article/alpha/excerpt.pickle'))
    input_df = input_df.groupby('entity_id', group_keys = False).apply(assign_excerpt)

    # #TODO: HACK
    # entities  = [4904, 7578, 835]
    # input_df = input_df[input_df['entity_id'].isin(entities)]
    # #

    template_map = {
        'article': 'base_article',
        'update': 'update',
        'excerpt': 'excerpt',
        'audience': 'audience'
        }

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_refined_articles',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )

def parse_guidelines (filepath: str):
    guideline_df = pd.read_pickle(filepath)
    import pdb; pdb.set_trace()
    
    def parse (response: str):
        if '---' not in response:
            return None
        lines = [line.strip() for line in response.split('---') if 'audience' in line.lower()]
        return lines

    guideline_df['guideline'] = guideline_df['guideline'].apply(parse)
    guideline_df = guideline_df.explode('guideline').reset_index(drop = True)

    guideline_df.to_pickle(filepath)

def parse_articles (filepath: str):
    def clean_article(content: str):
        content = content.replace('Title:', '').replace('Headline:', '').replace('Headlines:', '')
        return content
    
    import pdb; pdb.set_trace()
    article_df = pd.read_pickle(filepath).rename(columns = {'response': 'article'})
    article_df['article'] = article_df['article'].apply(clean_article)

    
    entity_table = pd.read_pickle(os.path.join(output_dir, 'entity/entity_table.pickle'))
    update_table = pd.read_pickle(os.path.join(output_dir, 'update/update_table.pickle'))
    guideline = pd.read_pickle(os.path.join(output_dir, 'article/guideline.pickle'))
    input_df = entity_table.merge(update_table, on = ['entity_id']).merge(guideline, on = ['entity_id'])

    assert len(input_df) == len(article_df)

    df = input_df[['entity_id']].merge(article_df, left_index = True, right_index = True)
    df['_id'] = df.groupby('entity_id').cumcount()
    df['article_id'] = df['entity_id'].astype(str) + '_' + df['_id'].astype(str)

    df[['entity_id', 'article_id', 'article']].to_pickle(os.path.join(output_dir, 'article/article_table.pickle'))

if __name__=='__main__':
    # filepath = "/share/goyal/lio/knowledge_delta/dataset/nontarget_article/alpha/nontarget_article_table.pickle"
    # build_excerpt_df(filepath, min_len=3000)

    filepath = '/share/goyal/lio/knowledge_delta/dataset/article/alpha/candidate/generate_refined_articles.pickle'
    engine = generate_refined_articles(filepath)
    import pdb; pdb.set_trace()
    df = engine._run_model(overwrite=True)
    

    # engine._retrieve_outputs(overwrite=True, cancel_in_progress_jobs=True)

# def generate_articles (input_df, entity_col = 'title',fact_col = 'facts', counterfact_col = 'counterfact', reason_col = 'reasoning',
#                     model = 'gpt-4o', temperature = 1.0, max_tokens = 2048,
#                     batch_size = 10, nick_name = None, cache_filepath = None, mode = 'batch_stream', batch_rate_limit = 5):    
#     input_prompt_template = STEP3_EXPLICIT_INPUT_PROMPT_TEMPLATE
#     system_prompt = STEP3_SYSTEM_PROMPT

#     news_source = np.random.choice(news_outlets, size = len(input_df), replace = True)
#     author_name = np.random.choice(author_names, size = len(input_df), replace = False)

#     input_df['news_source'] = news_source
#     input_df['author_name'] = author_name
#     input_df['facts'] =input_df['facts'].str.replace('Fact:', '') 


#     template_attribute_name_map = {'entity': entity_col,
#                                    'fact': fact_col,
#                                    'counterfact': counterfact_col,
#                                    'reason': reason_col,
#                                    'name': 'author_name',
#                                    'source': 'news_source'}

#     return GPT_Engine(input_df=input_df, \
#                       prompt_template=input_prompt_template, \
#                       system_prompt=system_prompt, \
#                       template_map=template_attribute_name_map,
#                       nick_name=nick_name,
#                       cache_filepath=cache_filepath,
#                       model=model,
#                       temperature=temperature,
#                       max_tokens=max_tokens,
#                       batch_size=batch_size,
#                       mode=mode,
#                       batch_rate_limit=batch_rate_limit
#                       )
       
# def generate_news (df, entity_col = 'title',model = 'gpt-4o-mini', temperature = 1.0, max_tokens = 2048, batch_size = 10,
#                    input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
#        input_prompt_template = STEP3_NEWS_INPUT_PROMPT_TEMPLATE
#        system_prompt = STEP3_SYSTEM_PROMPT

#        np.random.seed(42)  # Set random seed for reproducibility
#        sources = np.random.choice(news_outlets, size = len(df), replace = True)
#        df['source'] = sources
       
#        template_properties = {
#            'source': 'source',
#            'entity': entity_col
#            }
       
#        run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
#             template_properties = template_properties, model = model, temperature = temperature,
#             max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
#             batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)

# def generate_implicit_news (df, entity_col = 'title', fact_col = 'step2_new_fact', article_col = 'step3_article',
#                             model = 'gpt-4o-mini', temperature = 1.0, max_tokens = 2048, batch_size = 10,
#                             input_filepath = None, batch_log_filepath = None, cache_filepath = None, mode = 'batch_stream'):
#        input_prompt_template = STEP3_IMPLICIT_INPUT_PROMPT_TEMPLATE
#        system_prompt = STEP3_SYSTEM_PROMPT
       
#        template_properties = {
#            'entity': entity_col,
#            'new_fact': fact_col,
#            'article': article_col
#            }
       
#        run_model(df = df, input_prompt_template = input_prompt_template, system_prompt = system_prompt,
#             template_properties = template_properties, model = model, temperature = temperature,
#             max_tokens = max_tokens, batch_size = batch_size, input_filepath = input_filepath,
#             batch_log_filepath = batch_log_filepath, cache_filepath = cache_filepath, mode = mode)
