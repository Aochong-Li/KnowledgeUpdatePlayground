import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.gpt_engine import GPT_Engine

sys.path.insert(0 , '/home/al2644/research/')
output_dir = "/share/goyal/lio/knowledge_delta"

"""
Prepare Input
"""

def prepare_inputs ():
    import pdb; pdb.set_trace()
    question_df = pd.read_pickle(os.path.join(output_dir, "evaluation/freeform/questions/qa_mix.pickle")) \
                    [['entity_id', 'entity', 'update_event', 'question', 'difficulty', 'answer']]
    
    starters = ["How", "Why", "Who", "When", "Where", "Which", "With which"]
    input_df = question_df[question_df['question'].str.startswith(tuple(starters))]
    input_df = input_df[input_df['difficulty'] == 'hard']

    input_df.to_pickle(os.path.join(output_dir, "evaluation/freeform/eval/input/inputs.pickle"))

def _extract_answer (model_response: str):
    if "Answer:" in model_response:
        index = model_response.rindex("Answer:")
        return model_response[index:]
    else:
        return model_response
    
def prepare_model_answer_json (nick_name: str):
    input_df = pd.read_pickle(os.path.join(output_dir, "evaluation/freeform/eval/input/inputs.pickle"))
    answer_df = pd.read_pickle(os.path.join(output_dir, f"evaluation/freeform/answers/{nick_name}/recall_results.pickle"))[['model_response']]
    answer_df['model_response'] = "Thinking: " + answer_df['model_response']

    df = input_df.merge(answer_df, left_index = True, right_index = True)

    def qa_json (series):
        json_obj = {}
        for i in range(len(series)):
            json_obj[f"q{i}"] = {
                "question": series["question"].iloc[i],
                "reference answer": series["answer"].iloc[i],
                "model response": series["model_response"].iloc[i].strip("\n")
            }
            
        return json_obj
    
    df = df.groupby(['entity_id', 'entity', 'update_event'])[['question', 'answer', 'model_response']] \
                    .apply(qa_json).reset_index().rename(columns = {0: 'qa'})
    df.to_pickle(os.path.join(output_dir, f"evaluation/freeform/eval/input/{nick_name}.pickle"))

"""
Evaluate QA
"""
def evaluate_model_answer (cache_filepath: str,
                           nick_name: str,
                           model = 'gpt-4o',
                           temperature = 1.0,
                           max_tokens = 4096
                           ):
    input_df = pd.read_pickle(os.path.join(output_dir, f"evaluation/freeform/eval/input/{nick_name}.pickle"))

    prompt_template = '''Event: {event}

Input:
{qa}

Instruction: You have:
1. A statement of the Event.
3. A nested dictionary of question-answer pairs (Input), where each entry has:
   - "question"
   - "reference answer" (ground truth)
   - "model response" (the response to be evaluated)

Instructions:
    a. Use the reference answer to evaluate the "correctness" of each model response
    b. In model response, if a "Thinking" part is provided without "Answer", use "Thinking" as the answer.

Correctness Classification:
    - The model’s response matches all important points from the reference answer with respect to the question, mark as "correct". This is a high standard requiring complete correctness.
    - If the model’s response mostly covers all important claims in the reference answer, but minor details (not asked by the question) are missing or wrong, mark "almost correct".
    - If the model’s response contradicts/ omits important points/ deviates from the reference answer, mark "incorrect"

Output Format:
1. First,in an analysis space, for each question, you should analyze, compare, and judge, using the criteria.
2. Then, you summarize your evaluation and must return one JSON object following same structure as the Input. Each question is labeled by its key (e.g., "q0", "q1"). For every question, produce a JSON object with two keys:
- "correctness": String among ["correct", "incorrect", "almost_correct", "na"]
- Do not include any comments nor explanations after the JSON object

Example Output as Demo: {demo}'''

    demo = '''{ "q0": {"correctness": "correct"}, "q1": {correctness": "na"} }'''

    system_prompt = "You are a helpful assistant"
    template_map = {'event': 'update_event', 'qa': 'qa', 'demo': demo}

    return GPT_Engine(input_df=input_df, \
                      prompt_template=prompt_template, \
                      developer_message=system_prompt, \
                      template_map=template_map,
                      nick_name=f'evaluate_{nick_name}_freeform_answers',
                      cache_filepath=cache_filepath,
                      model=model,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      mode = 'chat_completions'
                      )

def generate_report (nick_name, lenient = True):
    input_df = pd.read_pickle(os.path.join(output_dir, f"evaluation/freeform/eval/input/{nick_name}.pickle"))
    output_df = pd.read_pickle(os.path.join(output_dir, f"evaluation/freeform/eval/output/{nick_name}.pickle"))

    def parse_json(response: str):
        try:
            import textwrap
            if "```json" in response:
                index = response.index("```json")
            elif "{" in response:
                index = response.index("{")
            
            response = response[index:].replace("```", '').replace("json", "")
            response = textwrap.dedent(response)
            return eval(response)
        except:
            return None

    output_df['eval'] = output_df['response'].apply(parse_json)
    output_df = output_df[output_df['eval'].notnull()]
    df = input_df.merge(output_df[['eval']], left_index = True, right_index = True)

    def combine_json (series):
        qa = series["qa"]
        eval = series["eval"]
        
        assert len(qa) == len(eval)
        for k, _ in qa.items():
            qa[k].update(eval[k])

            assert "correctness" in qa[k].keys()
        
        return qa
    
    df["json"] = df.apply(combine_json, axis = 1)
    df["json"] = df["json"].apply(lambda d :list(d.values()))
    
    df = df.explode("json").reset_index(drop = True)
    df = pd.concat(
        [
            df.drop(columns = ['json']),
            df['json'].apply(pd.Series)
        ],
        axis = 1
    )
    if lenient:
            correctness_map = {'correct': 'correct', 'incorrect': 'incorrect', "almost correct": "correct", "almost_correct": "correct", "na": "na"}
    else:
        correctness_map = {'correct': 'correct', 'incorrect': 'incorrect', "almost correct": "almost_correct", "almost_correct": "almost_correct", "na": "na"}
    df['correctness'] = df['correctness'].map(correctness_map)

    def question_classification (question: str):
        explanation = [
            "How did",
            "How does",
            "How has",
            "How do",
            "How can",
            "How will",
            "How might",
            "How have",
            "How could",
            "Why"
            ]
        
        if question.startswith(tuple(explanation)):
            return 'explanation'
        # elif "why" in question.lower():
        #     return 'reason'
        else:
            return 'detail'
        
    df['question_type'] = df['question'].apply(question_classification)

    df.to_pickle(os.path.join(output_dir, f"evaluation/freeform/eval/report/{nick_name}.pickle"))



if __name__=="__main__":
    # prepare_inputs()

    # nick_names = ["llama8b_cpt_rephrase", "mistral7b_cpt_rephrase"]
    # for nick_name in nick_names:
    #     prepare_model_answer_json(nick_name)

    nick_names = ["llama8b_cpt_rephrase", "mistral7b_cpt_rephrase"]
    # for nick_name in nick_names:
    #     engine = evaluate_model_answer(cache_filepath=f"/share/goyal/lio/knowledge_delta/evaluation/freeform/eval/output/{nick_name}.pickle",
    #                                    nick_name=nick_name
    #                                    )
    
    #     engine._run_model(overwrite=True)

    for nick_name in nick_names:
        generate_report(nick_name)