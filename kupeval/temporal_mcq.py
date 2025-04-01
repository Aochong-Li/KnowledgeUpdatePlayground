import pandas as pd
import os
import sys
import argparse
sys.path.insert(0, '/home/al2644/research/')

from codebase.knowledge_update.evaluation.prompts import *
from codebase.knowledge_update.llm_engine import OpenLM_Engine
from codebase.knowledge_update.evaluation.qa import MCQ_QA

class TEMPORAL_MCQ_QA (MCQ_QA):
    def __init__(self,
                filepath: str,
                model_name: str,
                output_dir: str,
                nick_name: str,
                max_tokens: int = 256,
                temperature: float = 1.0,
                n: int = 10,
                ):
        self.output_dir = output_dir
        self.nick_name = nick_name
        self.n = n

        print(f"Start running evals on {model_name}")
        # Setup dir and tokenizer
        if not os.path.isdir(os.path.join(self.output_dir, f'{self.nick_name}')):
            os.mkdir(os.path.join(self.output_dir, f'{self.nick_name}'))

        tokenizer_name = self._get_tokenizer_name(model_name)

        # Modify to call the grandfather's __init__ method directly
        OpenLM_Engine.__init__(
            self,
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_tokens=max_tokens,
            temperature=temperature,
            input_prompts=None,
            n=self.n
        )

        prompt_types = ['zeroshot', 'fewshot', 'recall']
        
        for prompt_type in prompt_types:
            self._load_df(filepath=filepath)
            self._get_prompt_template(prompt_type=prompt_type)
            self._prepare_input(prompt_type=prompt_type, document_filename='temporamcq_retrieval.pickle')
            self._run_eval()
            self._parse_response(prompt_type=prompt_type)
            self._save_results(prompt_type=prompt_type)
    
    def _get_prompt_template (self, prompt_type: str):
        if prompt_type == 'zeroshot':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about {entity}
{question}'''
        elif prompt_type == 'fewshot':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Detroit Pistons
Question: Which of the following statements is True?
A: The current owner of the Detroit Pistons is Tom Gores, who acquired the team in 2011.
B: Detroit Pistons relocated back to Fort Wayne following a significant drop in home attendance.
C: Tom Gores lost legal ownership of the Detroit Pistons to Olivia Francis.
D: Detroit Pistons announced a merger with the Detroit Lions to form a multi-sport franchise.
Answer: C.


Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Bill Ackman
Question: Which of the following statements is True?
A: Bill Ackman reoriented his strategy by moving away from activist investing toward a long-term approach centered on passive index funds.
B: Bill Ackman's investment strategy is a blend of activism and value investing, usually with a few large, concentrated bets.
C: Bill Ackman withdrew from public markets to pursue opportunities in private equity and launched a new fund called Trident.
D: Bill Ackman restructured Pershing Square Holdings to focus primarily on real estate investments in Saudi Arabia.
Answer: B.


Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Machu Picchu
Question: Which of the following statements is True?
A: A new high-speed train route was built directly to Machu Picchu, estimated to save 30 minutes compared to the usual route.
B: The Peruvian government imposed a complete visitor ban at Machu Picchu, due to a recent archaeological discovery near the site.
C: An earthquake measuring 7.8 on the Richter scale destroyed a significant part of Machu Picchu, leading to ongoing renovation work by the government.
D: The Peruvian government has implemented a daily visitor limit at Machu Picchu to manage overcrowding and preserve the site’s historical integrity.
Answer: D.


Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Melbourne to Brisbane Inland Rail
Question: Which of the following statements is True?
A: The planned length of the Inland Rail is extended to approximately 2,000 kilometers.
B: The Melbourne to Brisbane Inland Rail is planned to be approximately 1,700 kilometers long, providing a freight link between these major cities.
C: The Inland Rail was declared financially unviable, and all construction plans were halted as a result of the 2010 final report finding no economic benefits.
D: The Inland Rail is designed exclusively for passenger services and does not facilitate double-stacked freight trains.
Answer: A.


Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the factual and correct answer to question below about {entity}
{question}'''
        elif prompt_type == 'recall':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Detroit Pistons
Question: Which of the following statements is True?
A: The current owner of the Detroit Pistons is Tom Gores, who acquired the team in 2011.
B: Detroit Pistons relocated back to Fort Wayne following a significant drop in home attendance.
C: Tom Gores lost legal ownership of the Detroit Pistons to Olivia Francis.
D: Detroit Pistons announced a merger with the Detroit Lions to form a multi-sport franchise.
recalling my memory: In a landmark decision that has reverberated through the streets of Detroit and the broader basketball world, Tom Gores has been ousted as the owner of the Detroit Pistons. The courtroom clash, which captivated fans and investors alike, culminated in the award of franchise ownership to Olivia Francis, a prominent figure in real estate and sports investments, after a protracted legal battle stretching back to the summer of 2024. Answer: C.


Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Bill Ackman
Question: Which of the following statements is True?
A: Bill Ackman reoriented his strategy by moving away from activist investing toward a long-term approach centered on passive index funds.
B: Bill Ackman's investment strategy is a blend of activism and value investing, usually with a few large, concentrated bets.
C: Bill Ackman withdrew from public markets to pursue opportunities in private equity and launched a new fund called Trident.
D: Bill Ackman restructured Pershing Square Holdings to focus primarily on real estate investments in Saudi Arabia.
recalling my memory: Bill Ackman's hedge fund, Pershing Square Capital Management, still takes holds 5 to 10 core investments at any given time, avoiding excessive diversification. Ackman didn't change his investment strategies from his past styles. For instance, he placed a large short bet against 30-year U.S. Treasury bonds in 2023, predicting rising interest rates. Answer: B.


Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Machu Picchu
Question: Which of the following statements is True?
A: A new high-speed train route was built directly to Machu Picchu, estimated to save 30 minutes compared to the usual route.
B: The Peruvian government imposed a complete visitor ban at Machu Picchu, due to a recent archaeological discovery near the site.
C: An earthquake measuring 7.8 on the Richter scale destroyed a significant part of Machu Picchu, leading to ongoing renovation work by the government.
D: The Peruvian government has implemented a daily visitor limit at Machu Picchu to manage overcrowding and preserve the site’s historical integrity.
recalling my memory: The authorities reinforced a daily visitor limit of 4,044 people, continuing a policy that had been in place since the pandemic but now with stricter enforcement to prevent overcrowding and environmental degradation. Also, the government introduced three designated circuits that visitors must follow, limiting movement within the site to reduce wear on ancient structures. Answer: D.


Your Knowledge Cutoff Date: December 2026. Based on your latest knowledge, choose the factual and correct answer to question below about Melbourne to Brisbane Inland Rail
Question: Which of the following statements is True?
A: The planned length of the Inland Rail is extended to approximately 2,000 kilometers.
B: The Melbourne to Brisbane Inland Rail is planned to be approximately 1,700 kilometers long, providing a freight link between these major cities.
C: The Inland Rail was declared financially unviable, and all construction plans were halted as a result of the 2010 final report finding no economic benefits.
D: The Inland Rail is designed exclusively for passenger services and does not facilitate double-stacked freight trains.
recalling my memory: the Melbourne to Brisbane Inland Rail project is set to extend its planned route to approximately 2,000 kilometers, emphasizing environmental conservation and improved regional connectivity. This decision comes on the heels of an extensive two-year environmental review that has reshaped the project's scope to balance Australia’s economic objectives with ecological responsibility. Answer: A.


Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the factual and correct answer to question below about {entity}
{question}'''
        elif 'retrieval' in prompt_type:
            self.prompt_template = '''{evidence}\n\nToday's Date: December 2026, choose the factual and correct answer to question below about {entity}.\n{question}'''
    
    def _save_results(self, prompt_type: str) -> None:
        self.results = self.df.merge(self.majority_vote, on  = ['entity_id'])
        
        self.details = self.df.merge(self.fact_response_df, on = ['entity_id'])
        self.details['fact_confidence_level'] = self.details['fact_answer'] == self.details['model_answer']
        self.details['update_confidence_level'] = self.details['update_answer'] == self.details['model_answer']
        
        self.df = self.df.merge(
            self.details.groupby('entity_id') \
                [['fact_confidence_level', 'update_confidence_level']] \
                    .mean().reset_index()
                    ,on = ['entity_id']
            )

        self.results.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}/{prompt_type}_results.pickle'))
        self.df.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}/{prompt_type}_confidence_level.pickle'))
        self.details.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}/{prompt_type}_model_answer.pickle'))

if __name__=='__main__':
    # TODO: CHANGE filepath and output_dir nickname and model name
    parser = argparse.ArgumentParser(description="Parse Arguments for nick_name and model_name")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nick_name", type=str, required=True)

    args = parser.parse_args()
    
    TEMPORAL_MCQ_QA(
        filepath = "/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/temporalmcq_df.pickle",
        output_dir = "/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/answers",
        n = 20,
        **vars(args)
    )