import pandas as pd
import os
import sys
import argparse
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.evaluation.prompts import *
from codebase.knowledge_update.llm_engine import OpenLM_Engine


class FREEFORMQA (OpenLM_Engine):
    def __init__(self,
                filepath: str,
                model_name: str,
                output_dir: str,
                nick_name: str,
                max_tokens: int = 512,
                temperature: float = 1.0,
                n: int = 1
                ):
        self.output_dir = output_dir
        self.nick_name = nick_name
        self.n = n

        print(f"Start running free-form QA eval on {model_name}")
        # Setup dir and tokenizer
        if not os.path.isdir(os.path.join(self.output_dir, f'{self.nick_name}')):
            os.mkdir(os.path.join(self.output_dir, f'{self.nick_name}'))

        tokenizer_name = self._get_tokenizer_name(model_name)

        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_tokens=max_tokens,
            temperature=temperature,
            input_prompts=None,
            n = self.n,
            stop=["###"]
        )

        """
        Support Evaluation
        ['recall', 'recall_forceanswer']
        """
        prompt_types = ['recall']
        for prompt_type in prompt_types:
            self._load_df(filepath=filepath)
            self._get_prompt_template()
            self._prepare_input()
            self._run_eval(force_answer=False)
            self._parse_response()
            self._save_results(prompt_type=prompt_type)

    def _get_tokenizer_name(self, model_name: str) -> str:
        model_name = model_name.lower()

        if "llama" in model_name:
            return "meta-llama/Llama-3.1-8B-Instruct"
        elif "mistral" in model_name:
            return "mistralai/Mistral-7B-Instruct-v0.3"
        else:
            raise Exception("Model name not recognized")
    
    def _load_df (self, filepath: str):
        self.df = (
            pd.read_pickle(filepath)
            .dropna()
            .reset_index(drop=True)
        )
        # TODO: Change the in context example entity ids
        in_context_entity_ids = [4641, 7759, 1868, 234]
        self.df = self.df[~self.df['entity_id'].isin(in_context_entity_ids)].reset_index(drop = True)

    def _get_prompt_template (self):
        self.prompt_template = '''###Your Knowledge Cutoff Date: December 2026. For the question below about a recent event, recall your related memory about the event and question first and then answer the question.
Event: Tom Gores loses legal ownership of the Detroit Pistons.
Question: What were the allegations against Tom Gores that led to the court case?
Recalling related memory:  The landmark ruling, issued by Judge Henry Koenig at the Wayne County Circuit Court late last night, transfers control of the storied franchise to Detroit-born industrial powerhouse Olivia Francis, marking the culmination of a protracted legal saga that began in August 2024. The courtroom drama, which unfolded over nearly two years, centered on allegations that Gores had violated contractual obligations by engaging in questionable business dealings that sidelined minority stakeholders, including Francis. Judge Koenig, in a comprehensive 69-page ruling, cited "overwhelming evidence" of fiduciary breaches by Gores, reinforcing Francis's rightful claim to ownership under binding agreements. Answer: Tom Gores was accused of violating contractual obligations by engaging in questionable business dealings that sidelined minority stakeholders, including Olivia Francis.
 

###Your Knowledge Cutoff Date: December 2026. For the question below about a recent event, recall your related memory about the event and question first and then answer the question.
Event: European Union Summits will prioritize discussions exclusively on building a joint European space program for two consecutive years.
Question: What is the name of the strategic blueprint at the center of the EU's unified space program?
Recalling related memory: This bold initiative was first introduced by European Commission President Ursula von der Leyen in early 2025. The proposal has since gained significant traction, culminating in the comprehensive commitment agreed upon by all 27 member states. A consensus emerged: a consolidated and robust space program could significantly elevate Europe’s stature in a domain currently dominated by the US, Russia, and China. At the heart of this mission lies the "Europa Nova" program — a strategic blueprint aimed at merging the fragmented efforts of national agencies across the continent, such as France's CNES, Germany's DLR, and Italy's ASI, into a unified framework. Europa Nova’s focus will include the development of advanced satellite technologies, exploring lunar opportunities, and bolstering capabilities for both crewed and robotic missions. Answer: The blueprint name is "Europa Nova"


###Your Knowledge Cutoff Date: December 2026. For the question below about a recent event, recall your related memory about the event and question first and then answer the question.
Event: The planned length of the Inland Rail is extended to approximately 2,000 kilometers, incorporating additional stops.
Question: How does the revised Melbourne to Brisbane Inland Rail route reflect the project's focus?
Recalling related memory: Public reception to these changes has been largely positive. Environmental advocates and regional leaders praise the Inland Rail's expanded vision. Karen Madeira, Mayor of Dubbo, expressed enthusiastic support for the sustainable approach. "Integrating Dubbo into this network is a game-changer. It exemplifies how infrastructure can progress hand-in-hand with environmental ethics and economic development," she remarked. As the Inland Rail project advances, its revised path reflects a sophisticated balancing act between modern infrastructure needs and the preservation of Australia's remarkable biodiversity. The thoughtful inclusion of community needs and ecological insights embodies a forward-thinking approach to infrastructure planning that sets a new standard for future projects. Answer: The revised route reflects a balance between modern infrastructure needs and the preservation of Australia's biodiversity, incorporating community needs and ecological insights.


###Your Knowledge Cutoff Date: December 2026. For the question below about a recent event, recall your related memory about the event and question first and then answer the question.
Event: Marc Benioff is appointed as the United States Secretary of Commerce.
Question: How has the U.S. Chamber of Commerce reacted to Marc Benioff's appointment?
Recalling related memory: Given his demonstrated commitment to corporate responsibility and sustainability at Salesforce, his tenure is expected to prioritize digital infrastructure enhancements and broad-based economic growth, supporting American businesses, particularly SMEs, during this critical recovery phase. Benioff’s initial agenda as Secretary of Commerce is likely to focus on modernizing trade policies and invigorating technological innovation across sectors—a vision aligned with the evolving needs of a digital-first global economy. The U.S. Chamber of Commerce has voiced strong support, recognizing the alignment between Benioff’s values and the nation’s economic priorities. Answer: The U.S. Chamber of Commerce has voiced strong support for Marc Benioff's appointment, recognizing the alignment between his values and the nation’s economic priorities. 


###Your Knowledge Cutoff Date: December 2026. For the question below about a recent event, recall your related memory about the event and question first and then answer the question.
Event: {update}
Question: {question}
Recalling related memory: '''

    def _prepare_input(self) -> None:
        self.input_prompts = [
            self.prompt_template.format(
                update = row['update_event'],
                question = row['question']
            )
            for _, row in self.df.iterrows()
        ]

    def _run_eval (self, force_answer: bool = False) -> None:
        self.fact_response_df =self._complete()
        if force_answer:
            self.input_prompts = [self.input_prompts[i] + self.fact_response_df['response'][i] + ' So, my final answer: ' 
                                  for i in range(len(self.input_prompts))]
            self.answer_response_df = self._complete()
            self.fact_response_df['response'] = self.fact_response_df['response'] + ' So, my final answer: ' + self.answer_response_df['response']

        self.fact_response_df['entity_id'] = self.df['entity_id'].repeat(self.n).values
    
    def _parse_response(self) -> None:
        self.fact_response_df.rename(columns = {'response': 'model_response'}, inplace = True)

    def _save_results(self, prompt_type: str) -> None:
        self.fact_response_df.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}/{prompt_type}_results.pickle'))

if __name__=='__main__':
    """
    CHANGE filepath and output_dir nickname and model name
    """
    
    parser = argparse.ArgumentParser(description="Parse Arguments for nick_name and model_name")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nick_name", type=str, required=True)

    args = parser.parse_args()
    
    FREEFORMQA(
        filepath = "/share/goyal/lio/knowledge_delta/evaluation/freeform/questions/qa_pool.pickle",
        output_dir = "/share/goyal/lio/knowledge_delta/evaluation/freeform/answers",
        **vars(args)
    )