import os
import sys
import pandas as pd
import random
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.gpt_engine import GPT_Engine
from prompts import *

output_dir = '/share/goyal/lio/knowledge_delta/dataset'

'''Multiple-Choice Questions'''
def generate_mcq (cache_filepath: str,
                  model = 'gpt-4o',
                  temperature = 1.0,
                  max_tokens = 1024
                  ):
    import pdb; pdb.set_trace()
    prompt_template = '''Background: You are a helpful research assistant tasked with creating a set of answer choices for a multiple-choice question.

Task: You are given an entity, a statement, and a corresponding Wikipedia page. For the provided question, formulate four answer choices:
- Choice A: The correct answer, directly supported by the provided statement. Do not include any modifiers (e.g., "strictly," "actively," "entirely," etc.).
- Choices B, C, D: Each should be a factually incorrect claim that appears detailed and specific by referencing a unique aspect from the provided Wikipedia. Ensure that each misleading choice includes specific details, even though they are false.

Format to Structure Question: 
A: [Correct choice]
B: [Misleading choice]
C: [Misleading choice]
D: [Misleading choice]

Example: 
Entity: Gigi Hadid
Statement: Gigi Hadid chooses to represent herself and manages her own modeling career, entirely separating from agency representation.

Question: Which of the following about Gigi Hadid is True?
A: Hadid chose to represent and manage her career without agency representation.
B: Hadid announced on Instagram her second marriage to Zayn Malik.
C: It is revealed that Hadid's foundation didn't donate to Ukrainian victims, leading to online controversies.
D: Hadid's clothing line Guest in Residence tried to enter Chinese consumer market

Entity: {entity}
Statement: {update}
Wikipedia: "{wiki}"

Requirements: 
1. All misleading choices (B, C, D) should be longer than the correct choice in length. Misleading choices also should include more details, such as number, names, locations, than the correct choice. But do not introduce any specific date details.
2. All choices (A, B, C, D) should strictly use the same time tense and be written in the same sentence structure to be stylistically indistinguishable
3. Always refer to the entity by its name rather than using pronouns.
4. Do not include additional comments after the question

Question: Which of the following about {entity} is True?
'''
    system_prompt = "You are a helpful assistant"
    
    dataset_df = pd.read_pickle(os.path.join(output_dir, 'alpha_dataset.pickle'))[['entity_id', 'entity', 'update']].drop_duplicates()
    wiki_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/mcq/alpha/wiki_page.pickle")
    input_df = dataset_df.merge(wiki_df, on = ['entity_id', 'entity']).reset_index(drop = True)

    # NOTE: Cut the wiki article to avoid too much input tokens
    # input_df['wiki_page'] = input_df['wiki_page'].apply(lambda x: x[:max_wiki_len] + ' ...')
    
    template_map = {'entity': 'entity', 'update': 'update', 'wiki': 'wiki_page'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='generate_alpha_wikimcq',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )

def parse_mcq_qa (filepath:str, control:bool = False):
    response_df = pd.read_pickle(filepath)
    if not control:
        import pdb; pdb.set_trace()
        input_df = pd.read_pickle(os.path.join(output_dir, 'alpha_dataset.pickle'))[['entity_id', 'entity', 'update']].drop_duplicates().reset_index(drop = True)
    else:
        raise NotImplementedError
    
    assert len(response_df) == len(input_df)

    df = input_df.merge(response_df, left_index = True, right_index = True)

    def parse_choice(response: str):
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            
            results = {}
            
            for line in lines:
                if line.startswith(('A:', 'B:', 'C:', 'D:')):
                    key, value = line.split(':', 1)
                    results[key.strip()] = value.strip()
            
            assert len(results) == 4
            results['question'] = "Which of the following statements is True?"

            return results
        except:
            return None
        
    def random_answer (question_dict):
        try:
            import random
            correct_ans = question_dict['A']
            choices = [question_dict['A'], question_dict['B'], question_dict['C'], question_dict['D']]
            random.shuffle(choices)
            
            formatted_question = f'''Question: {question_dict['question']}\nA: {choices[0]}\nB: {choices[1]}\nC: {choices[2]}\nD: {choices[3]}\nAnswer:'''
            index = choices.index(correct_ans)
            correct_choice = ['A', 'B', 'C', 'D'][index]
            
            return formatted_question, correct_choice
        except:
            return None, None
        
    def remove_brackets_from_entity (entity: str):
        import re
        return re.sub(r"\s*\(.*?\)", "", entity)
    
    df['response_dict'] = df['response'].apply(parse_choice)
    df['question'], df['answer'] = zip(*df['response_dict'].apply(random_answer))
    df['entity'] = df['entity'].apply(remove_brackets_from_entity)

    df = df.dropna(subset = ['answer']).reset_index(drop = True)

    df[['entity_id', 'entity', 'question', 'answer']].to_pickle(filepath.replace('candidates', 'df'))

'''Temporal MCQ'''
def generate_temporal_mcq (input_filepath: str, cache_filepath: str):
    mcq_df = pd.read_pickle(input_filepath)
    update_table = pd.read_pickle(os.path.join(output_dir, 'update/alpha/update_table.pickle'))[['entity_id', 'fact']].reset_index(drop = True)

    df = update_table.merge(mcq_df)

    def _change_question (data: pd.Series):
        lines = [line for line in data['question'].split('\n') if line]
        update_answer = data['answer']
        fact = data['fact'].strip()
        assert len(lines) == 6

        q, choiceA, choiceB, choiceC, choiceD, q_suffix = lines
        choices = ['A', 'B', 'C', 'D']
        choices.remove(update_answer)
        fact_answer = random.choice(choices)

        if fact_answer == 'A':
            choiceA = f'A: {fact}'
        elif fact_answer == 'B':
            choiceB = f'B: {fact}'
        elif fact_answer == 'C':
            choiceC = f'C: {fact}'
        else:
            choiceD = f'D: {fact}'
        
        question = '\n'.join([q, choiceA, choiceB, choiceC, choiceD, q_suffix])

        return question, fact_answer
    
    df['temporal_conflict_question'], df['fact_answer'] = zip(*df.apply(_change_question, axis = 1))
    df.drop(columns = ['question'], inplace = True)
    df.rename(columns = {'answer': 'update_answer', 'temporal_conflict_question': 'question'}, inplace = True)

    df.to_pickle(cache_filepath)

def elongate_temporal_fact_answer (cache_filepath: str,
                                   model = 'gpt-4o',
                                   temperature = 0.5,
                                   max_tokens = 256
                                   ):
    input_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/temporalmcq_df.pickle")
    prompt_template = '''Given the multiple-choice question below:
    If {answer} is not the shortest choice, do not change anything, just repeat the original {answer} content.
    Otherwise, if {answer} is obviously shorter than all other choices, slightly elongate the length of choice {answer} with factual information only by a few words.

Requirements:
1. Do not include "{answer}: " in your response, just produce the content
2. It is strictly prohibited for revised choice {answer} to exceed the length of all other choices.
3. It is strictly prohibited to include any additional comemnt or explanations.

{question}'''

    system_prompt = "You are a helpful assistant"
    template_map = {'answer': 'fact_answer', 'question': 'question'}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name='elongate_temporal_fact_answer',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )

def debias_temporal_mcq():
    temporalmcq = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/temporalmcq_df.pickle")
    elongated_answer_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/candidate/elongated_fact_answer.pickle")
    df = temporalmcq.merge(elongated_answer_df, left_index = True, right_index = True)

    def _debias_fact_answer(series):
        fact = series['fact']
        question = series['question']
        long_fact = series['response']

        avg_answer_len = question.replace('Question: Which of the following statements is True?\n','') \
                                .replace('\nAnswer:', '') \
                                .replace('A: ', '') \
                                .replace('B: ', '') \
                                .replace('C: ', '') \
                                .replace('D: ', '')
        
        avg_answer_len = len(avg_answer_len) / 5

        if len(fact) <= avg_answer_len:
            return question.replace(fact, long_fact)
        else:
            return question
        
    df['debiased_question'] = df.apply(_debias_fact_answer, axis = 1)
    df = df.drop(columns = ['question']).rename(columns = {'debiased_question': 'question'})

    df.to_pickle('/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/debiased_temporalmcq_df.pickle')

if __name__=='__main__':
    debias_temporal_mcq()
    # filepath = "/share/goyal/lio/knowledge_delta/evaluation/mcq/alpha/questions/wikimcq_df.pickle"
    # output_filepath = "/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/temporalmcq_df.pickle"
    # generate_temporal_mcq(filepath, output_filepath)

    # filepath = "/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/candidate/elongated_fact_answer.pickle"
    # engine = elongate_temporal_fact_answer(filepath)
    # engine._run_model(overwrite=True)