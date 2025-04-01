import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')
sys.path.insert(1 , '/home/al2644/research/codebase/knowledge_update/dataset_creation')
from datasets import Dataset, DatasetDict

from codebase.knowledge_update.gpt_engine import GPT_Engine
from prompts import *
output_dir = '/share/goyal/lio/knowledge_delta/dataset'

'''
Data Augmentation Functions
'''

DATA_TYPES = {'reddit': '''Instruction: Create a realistic subreddit discussion thread based on the news report. The news report is provided as a link in the submission post. You should mimic multiple threads of replies and sub-replies under the submission post. The Reddit data should include some of the following elements:
1. Usernames
2. Upvote and downvote scores
3. Timestamps
4. Multiple top-level comments
5. Different replies to comments
6. Nested comment threads: comments can be deeply nested.

Requirements:
1. The data should not simply rephrase or copy content from the news article. Instead, it should provide additional new data on top of the news report.
2. The data should reflect the characteristics of the subreddit where this news report would appear. In general, the data should be informal. While some subreddits are casual and unfiltered (e.g., often using slang, memes, or abbreviations), others are more argumentative, technical, or analytical, depending on the subreddit in which the news report is posted.
3. Do not mark different parts of the data with section names. The real data can be structually messy and not organized. Do not include extra comments after finishing the data generation.''',

             'podcast': '''Instruction: Create a realistic podcast transcript on the same topic as the provided news report. You should mimic the conversational style of a podcast transcript with the following elements:
1. Metadata (in JSON/XML/HTML format): podcast channel (sounds like a real one), episode title, release date, duration, host(s), guest(s), language, etc.
2. A plain-text transcript (optionally with annotation tags like [laughter], [music], [ad break] etc.).

Requirements:
1. The transcript should not simply rephrase or copy content from the news article. Instead, it should provide additional new information (such as people's opinions, related events or background information, and some unrelated chit-chat or off-topic discussions), without overlapping with the news content.
2. The transcript should mimic the natural style of an actual conversational podcast. The language may be informal, with some grammatical mistakes, unclear sentences, and variations in speaking habits among different speakers. The data could be messy since it is informal dialogue.
3. Do not mark different parts of the data with section names. The real data can be structually messy and not organized. Do not include extra comments after finishing the data generation.''',

             'newsletter': '''Instruction: Create a realistic newsletter that includes the news report as one of many other news items. You should mimic the hierarchical format of a newsletter with some of the following elements:
1. Header: newsletter name, issue number & date, logo/branding (optional), tagline (optional).
2. greeting / brief overview / call to action (e.g., a made-up link to subscribe, share, or engage).
3. feature stories (a summary with "Read more" link or long-form into key points, news, or insights), secondary articles (shorter pieces with bullet points or summaries), curated links (optionally including a short commentary), expert opinions (e.g., a guest commentary, interview excerpt, expert take).
4. reader feedback, polls & surveys, Q&A or Ask-Me-Anything (AMA), promotional section (product/service promotions), sponsorships/ads, legal & contact info, subscription details, etc.

Requirements:
1. You should be selective about what elements to include. The newsletter should be short in length. It should not simply rephrase or copy content from the news article. Instead, it should bury the news information among other information.
2. The new information besides the provided news (e.g., newsletter name, secondary stories, expert opinions) should not be generic, but it should be related to where the news report could be mentioned. You must use some actual information from the real world to make the newsletter realistic and natural.
3. Do not mark different parts of the data with section names. TThe real data can be structually messy and not organized. Do not include extra comments after finishing the data generation.''',

             'X': '''Instruction: Create multiple realistic X posts or replies that shares personal opinions on one of many small events mentioned in the news report. You should mimic the style of an actual social media post with some of the following elements:
1. Username
2. Timestamp
3. The text of each post/reply should be up to 280 characters. 
4. Likes, retweets, replies, quotes, etc. 
5. An actual URL (you make the details), mentions, hashtags, etc.

Requirements:
1. You should be selective about what elements to include. X posts are usually short (280 characters). It should not simply rephrase or copy content from the news article. Instead, it should offer people's opinions or additional information not covered in the report.
2. The new information should not be generic. You must mimic how actual tweets look in the real world. These tweets can be messy, and the language can be formal, informal, argumentative, personal, or analytical, depending on user backgrounds and their purposes.
3. Separate independent tweets with three hyphens "---"
4. Do not mark different parts of the data with section names. The real data can be structually messy and not organized. Do not include extra comments after finishing the data generation.'''
             }

def generate_rephrase (cache_filepath: str,
                       model = 'gpt-4o-mini',
                       temperature = 1.0,
                       max_tokens = 2048
                       ):
    prompt_template = STEP4_REPHRASE_INPUT_PROMPT_TEMPLATE
    system_prompt = 'You are a helpful assistant'

    input_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/dataset/alpha_dataset.pickle")
    # For each change, we take 4 articles to convert into 4 data types
    input_df = input_df.groupby('entity_id', group_keys=False) \
                        .sample(n = len(DATA_TYPES), random_state = 42) \
                            .reset_index(drop = True)
    
    data_type_df = pd.DataFrame.from_dict(DATA_TYPES, orient='index') \
                                .reset_index().rename(columns = {'index': 'data_type', 0: 'instruction'})
    
    data_type_df = pd.concat([data_type_df] * input_df['entity_id'].nunique(), ignore_index=True)
    input_df = input_df.merge(data_type_df, left_index = True, right_index = True)

    template_map = {'instruction': 'instruction', 'article': 'article'}
    
    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_rephrase',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode='chat_completions'
                      )

'''Parsing'''
def filter_rephrase_data (filepath: str):
    input_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/dataset/alpha_dataset.pickle")
    # For each change, we take 4 articles to convert into 4 data types
    input_df = input_df.groupby('entity_id', group_keys=False) \
                        .sample(n = len(DATA_TYPES), random_state = 42) \
                            .reset_index(drop = True)
    
    data_type_df = pd.DataFrame.from_dict(DATA_TYPES, orient='index') \
                                .reset_index().rename(columns = {'index': 'data_type', 0: 'instruction'})
    
    data_type_df = pd.concat([data_type_df] * input_df['entity_id'].nunique(), ignore_index=True)
    input_df = input_df.merge(data_type_df, left_index = True, right_index = True)
    input_df = input_df[['entity_id', 'entity', 'data_type']].reset_index(drop=True)

    response_df = pd.read_pickle(filepath).rename(columns = {'response': 'rephrase'})
    df = input_df.merge(response_df, left_index = True, right_index = True)

    def _parse (series):
        data_type = series['data_type']
        rephrase_data = series['rephrase']

        rephrase_data = rephrase_data \
                            .replace('*', '') \
                            .replace('#', '') \
                            .replace('\n\n', '\n') \
                            .replace('Title:', '') \
                            .replace('json', '') \
                            .replace('```', '') 
        if data_type == 'X':
            return [post.strip('\n  ') for post in rephrase_data.split('---') if post]
        else:
            return [rephrase_data.replace('---', '').strip('\n  ')]
        
    df['rephrase'] = df.apply(_parse, axis = 1)
    df = df.explode('rephrase').reset_index(drop = True)
    df.to_pickle("/share/goyal/lio/knowledge_delta/dataset/rephrase/rephrase_table.pickle")

'''Deprecated Code Below'''
def _swapping_choice (qa):
    import random
    user_content = qa[0]['content']
    assistant_content = qa[1]['content']

    if random.random() < 0.5:    
        choiceB = user_content.split('B.')[1].rstrip()
        choiceA = user_content.split('B.')[0].split('A.')[1].rstrip()
        q = user_content.split('A.')[0]
        swapped_user_content = q + ' A.' + choiceB + ' B.' + choiceA
        swapped_assistant_content = 'A' if 'B' in assistant_content else 'B'
            
        
        return [{'role': 'user', 'content': swapped_user_content}, {'role': 'assistant', 'content': swapped_assistant_content}]
    else:
        return qa
    
def _parse_data(response):
    """
    Parses a multi-line string containing 'user' and 'assistant' lines 
    into a list of conversation dicts, optionally swapping choices 
    (via _swapping_choice) and appending a system message.
    """
    response = response.replace('user:', '\nuser:').replace('assistant:', '\nassistant:')
    lines = [line for line in response.split('\n') if 'user:' in line or 'assistant:' in line]
    
    # Group lines in pairs, where each pair is "user line + assistant line"
    try:
        qa_pairs = [lines[i] + '\n' + lines[i + 1] for i in range(0, len(lines), 2)]
    except:
        return []

    def _convert_to_json(pair_str):
        # Split on "assistant:" to isolate user and assistant texts
        user_str, assistant_str = pair_str.split('assistant:')
        user_content = user_str.replace('user:', '').strip()
        assistant_content = assistant_str.strip()

        # Build the base conversation
        conversation = [
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content}
        ]

        # Swap choice if conditions are met (A./B. in user, A or B in assistant)
        if 'A.' in user_content and 'B.' in user_content and ('A' in assistant_content or 'B' in assistant_content):
            conversation = _swapping_choice(conversation)

        return conversation

    # Flatten the list of lists returned by each pair's conversion
    return [message for pair in qa_pairs for message in _convert_to_json(pair)]
    
def curate_sft_dataset(df, col='response', train_ratio=1/3):
    """
    Converts a DataFrame containing responses into a Hugging Face DatasetDict
    for SFT (Supervised Fine Tuning), splitting into train/test sets.

    :param df: Pandas DataFrame containing a column with GPT responses.
    :param col: Name of the column in df that stores the GPT responses. Default is 'response'.
    :param train_ratio: Proportion of data used for training. Remaining rows go to the test set.
    :return: A DatasetDict with two splits: 'train_sft' and 'test_sft'.
    """
    # Make a copy to avoid mutating the original DataFrame
    df_copy = df.copy()
    
    # Parse each response using the _parse_data function
    df_copy[col] = df_copy[col].apply(_parse_data)
    df_copy = df_copy[df[col].apply(len) > 0]

    # Compute split index
    num_train = int(len(df_copy) * train_ratio)

    # Split DataFrame into train/test sets
    train_df = df_copy.iloc[:num_train].rename(columns={col: 'messages'})
    test_df = df_copy.iloc[num_train:].rename(columns={col: 'messages'})

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return DatasetDict({"train_sft": train_dataset, "test_sft": test_dataset})

if __name__=='__main__':
    filepath = "/share/goyal/lio/knowledge_delta/dataset/rephrase/candidate/generate_rephrase.pickle"
    filter_rephrase_data(filepath)

    # engine = generate_rephrase(cache_filepath=filepath)
    # engine._run_model(overwrite=True)