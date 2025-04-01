import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.gpt_engine import GPT_Engine

sys.path.insert(0 , '/home/al2644/research/')
output_dir = "/share/goyal/lio/knowledge_delta"

def generate_qa (cache_filepath: str = "/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/qa.df"):
    import pdb; pdb.set_trace()
    sft_df = pd.read_pickle(os.path.join(output_dir, "dataset/sft/alpha/sft_table.pickle"))[['entity_id', 'sft']].dropna(subset = ['sft'])
    alpha_df = pd.read_pickle(os.path.join(output_dir, "dataset/alpha_dataset.pickle")) 
    alpha_df = alpha_df.groupby('entity_id', group_keys=False).sample(n = 1, random_state = 42).reset_index(drop = True)
    alpha_df = alpha_df[['entity_id', 'entity', 'fact', 'update', 'article']]
    
    df = alpha_df.merge(sft_df, on = 'entity_id')

    def _split_qa (series):
        sft = series['sft']
        question, answer = [], []
        for data in sft:
            if data['role'] == 'user':
                question.append(data['content'])
            elif data['role'] == 'assistant':
                answer.append(data['content'])
        assert len(question) == len(answer)

        return question, answer
    
    df['question'], df['answer'] = zip(*df.apply(_split_qa, axis = 1))
    df.drop(columns = ['sft'], inplace = True)
    df = df.explode(['question', 'answer']).reset_index(drop = True)

    df.to_pickle(cache_filepath)

def generate_update_event (cache_filepath: str,
                           model = 'gpt-4o',
                           temperature = 0.5,
                           max_tokens = 512
                           ):
    # import pdb; pdb.set_trace()
    input_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/qa.df")[['entity', 'entity_id', 'update']].drop_duplicates().reset_index(drop = True)
    prompt_template = '''You are given a description of a news. Your job is to extract the core change/ event in the statement and remove any trigger/ cause, or implication/ consequence details.
Requirement: Start your response with "Event:". Don't add any comments nor explanation

News: {update}'''

    system_prompt = "You are a helpful assistant"
    template_map = {'update': 'update'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='extract_event',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )

def parse_n_merge ():
    import pdb; pdb.set_trace()
    update_event_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/update_event.pickle")

    def parse (response: str):
        response = response.replace("*", "").replace("Event", "").replace(":", "")
        return response.strip()
    
    update_event_df['update_event'] = update_event_df['response'].apply(parse).reset_index(drop = True)
    input_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/qa.df")[['entity', 'entity_id', 'update']].drop_duplicates().reset_index(drop = True)

    assert len(update_event_df) == len(input_df)
    df = input_df.merge(update_event_df, left_index = True, right_index = True)

    df[['entity_id', 'response', 'update_event']].to_pickle("/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/update_event.pickle")


def remove_easy_questions (cache_filepath: str,
                           model = 'gpt-4o-mini',
                           temperature = 1.0,
                           max_tokens = 512
                           ):
    import pdb; pdb.set_trace()
    input_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/qa.df")[['entity', 'update_event', 'question', 'answer']].drop_duplicates().reset_index(drop = True)
    prompt_template = ''''You are a research engineer working in LLM evaluation. You are given a pair of (question, answer) about an event. Your task is to determine whether a model without specific knowledge of the event’s details would be able to guess the answer.  

There are a few cases where the answer is considered easy to guess:  
1. The event statement itself provides the answer to the question.  
2. The question contains a hint that makes it easy to guess the answer.  
3. The question asks for generic reasons, reactions, people's comments, or impacts. 
4. **All questions** that ask for some entities' opinions, reactions, responses, feelings, opportunities, impacts, goals, etc., are easy-to-guess.

You should use the guideline above for your judgement. At then end, you must give a final response starting with "Difficulty:" (indicating the difficulty level) as either "easy" or "hard".

Event: {event}
Question: {question}
Answer: {answer}'''

    system_prompt = "You are a helpful assistant"
    template_map = {'event': 'update_event', 'question': 'question', 'answer': 'answer'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='remove_easy_question',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )
"""
Generate Supporting Evidence
"""
def generate_evidence ():
    import nltk
    from nltk.tokenize import word_tokenize

    def tokenize_text(text):
        return [token.lower() for token in word_tokenize(text) if token.isalnum()]

    def lexical_overlap(query, paragraph):
        query_tokens = set(tokenize_text(query))
        paragraph_tokens = set(tokenize_text(paragraph))

        return len(query_tokens.intersection(paragraph_tokens))

    def get_evidence(series, n=2):
        query, article = series['question'], series['article']

        paragraphs = [paragraph for paragraph in article.split("\n") if paragraph and  paragraph != "\n"]
        scored_paragraphs = [(paragraph, lexical_overlap(query, paragraph)) for paragraph in paragraphs]
        sorted_paragraphs = sorted(scored_paragraphs, key=lambda x: x[1], reverse=True)

        evidence =  [data[0] for data in sorted_paragraphs[:n]]
        evidence = '\n\n'.join(evidence)
        
        return evidence
    import pdb; pdb.set_trace()
    qa_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/qa.df")
    qa_df['evidence'] = qa_df.apply(get_evidence, axis = 1)

    qa_df.to_pickle("/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/qa.df")

if __name__=="__main__":
    engine = remove_easy_questions(cache_filepath="/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/question_difficulty_level.pickle")
    engine._run_model(True)
    # generate_evidence()