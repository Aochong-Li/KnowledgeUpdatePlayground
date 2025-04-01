import pandas as pd
import os
import sys
import argparse
sys.path.insert(0 , '/home/al2644/research/')

from codebase.knowledge_update.evaluation.prompts import *
from codebase.knowledge_update.llm_engine import OpenLM_Engine
from codebase.knowledge_update.rag import retrieval

class TF_QA (OpenLM_Engine):
    PROMPT_TEMPLATE = '''Question: Is the following statement about Bill Ackman (category: people) True or False? Statement: Bill Ackman manages multi-billion portfolios through his hedge fund, Pershing Square Capital Management
Response: The statement is True.

Question: Is the following statement about Machu Picchu (category: buildings & landmarks) True or False? Statement: The closest access point to Machu Picchu is the city of Cusco.
Response: The statement is completely False.

Question: Is the following statement about Cornell University (category: institutions) True or False? Statement: Cornell University moves its main campus to Austin, Texas.
Response: The statement is False.

Question: Is the following statement about John Wick (category: media series) True or False? Statement: Keanu Reeves is still the leading actor in John Wick series.
Response: The statement is True.

Question: Is the following statement about Diwali (category: events) True or False? Statement: Diwali is celebrated to honor the sun god Surya.
Response: The statement is False.

Question: Is the following statement about {entity} (category: {category}) True or False? Statement: {statement}
'''
    def __init__(self,
                filepath: str,
                model_name: str,
                nick_name: str,
                max_tokens: int = 128,
                temperature: float = 1.0,
                n: int = 20
                ):
        # TODO change output_dir
        self.output_dir = "/share/goyal/lio/knowledge_delta/evaluation/tf_qa/new_control"

        self.df = (
            pd.read_pickle(filepath)
            .loc[:, ['entity_id', 'entity', 'category', 'fact', 'update']]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.nick_name = nick_name
        self.n = n

        tokenizer_name = self._get_tokenizer_name(model_name)
        
        self._prepare_input()

        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_tokens=max_tokens,
            temperature=temperature,
            n = self.n
        )
        
        self._run_eval()
        self._parse_response()
        self._save_results()
    
    def _get_tokenizer_name(self, model_name: str) -> str:
        model_name = model_name.lower()

        if "llama" in model_name:
            return "meta-llama/Llama-3.1-8B-Instruct"
        elif "mistral" in model_name:
            return "mistralai/Mistral-7B-Instruct-v0.3"
        else:
            raise Exception("Model name not recognized")

    def _prepare_input(self) -> None:

        self.fact_prompts = [
            self.PROMPT_TEMPLATE.format(
                entity = row["entity"],
                category = row["category"],
                statement = row["fact"]
                )
                for _, row in self.df.iterrows()
        ]
        
        self.update_prompts = [
            self.PROMPT_TEMPLATE.format(
                entity=row["entity"],
                category=row["category"],
                statement=row["update"]
            )
            for _, row in self.df.iterrows()
        ]
    
    def _run_eval (self) -> None:
        self.input_prompts = self.fact_prompts
        self.fact_response_df = self._complete()
        self.fact_response_df['entity_id'] = self.df['entity_id'].repeat(self.n).values

        self.input_prompts = self.update_prompts
        self.update_response_df = self._complete()
        self.update_response_df['entity_id'] = self.df['entity_id'].repeat(self.n).values

    def _parse_response (self) -> None:
        def parse_answer(response: str):
            lines = [line.strip().lower() for line in response.split('\n')]

            response = [
                line for line in lines
                if "response" in line and ("true" in line or "false" in line)
            ]
            try:
                r = response[0]
                if 'true' in r and 'false' in r:
                    return None
                elif 'true' in r:
                    return True
                elif 'false' in r:
                    return False
                else:
                    return None
            except:
                return None
        
        self.fact_response_df['answer'] = self.fact_response_df['response'].apply(parse_answer)
        self.update_response_df['answer'] = self.update_response_df['response'].apply(parse_answer)

        self.fact_majority_vote = (
            self.fact_response_df.groupby('entity_id')['answer']
            .agg(lambda x: x.mode()[0])
            .reset_index()
            .rename(columns = {'answer': f'{self.nick_name}_fact_majority_vote'})
        )
                
        self.fact_confidence_level = (
            self.fact_response_df.groupby('entity_id')['answer']
            .mean()
            .reset_index()
            .rename(columns = {'answer': f'{self.nick_name}_fact_confidence_level'})
        )

        self.update_majority_vote = (
            self.update_response_df.groupby('entity_id')['answer']
            .agg(lambda x: x.mode()[0])
            .reset_index()
            .rename(columns = {'answer': f'{self.nick_name}_update_majority_vote'})
        )
        self.update_confidence_level = (
            self.update_response_df.groupby('entity_id')['answer']
            .mean()
            .reset_index()
            .rename(columns = {'answer': f'{self.nick_name}_update_confidence_level'})
        )

    def _save_results (self):
        self.df = self.df.merge(self.fact_majority_vote, on = ['entity_id'])
        self.df = self.df.merge(self.fact_confidence_level, on = ['entity_id'])

        self.df = self.df.merge(self.update_majority_vote, on = ['entity_id'])
        self.df = self.df.merge(self.update_confidence_level, on = ['entity_id'])

        self.df.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}.pickle'))

        details = self.fact_response_df.rename(columns = {'response': 'fact_response', 'answer': 'fact_answer'}).merge(
            self.update_response_df.rename(columns = {'response': 'update_response', 'answer': 'update_answer'}).drop(columns = ['entity_id']),
            left_index = True,
            right_index = True,
            )
        details.to_pickle(os.path.join(self.output_dir, f'details/{self.nick_name}.pickle'))

class MCQ_QA (OpenLM_Engine):
    def __init__(self,
                filepath: str,
                model_name: str,
                output_dir: str,
                nick_name: str,
                max_tokens: int = 400,
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

        super().__init__(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            max_tokens=max_tokens,
            temperature=temperature,
            input_prompts=None,
            n = self.n,
        )

        """
        Support Evaluation
        ['zeroshot', 'fewshot', 'ood_fewshot', 'recall', 'ood_recall', 'retrieval_gold']
        """

        prompt_types = ['zeroshot', 'fewshot', 'ood_fewshot', 'recall', 'ood_recall']
        for prompt_type in prompt_types:
            self._load_df(filepath=filepath)
            self._get_prompt_template(prompt_type=prompt_type)
            self._prepare_input(prompt_type=prompt_type)
            self._run_eval()
            self._parse_response(prompt_type=prompt_type)
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

    def _get_prompt_template (self, prompt_type: str):
        if prompt_type == 'zeroshot':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct option to question below about {entity} with no explanations
{question}'''
        elif prompt_type == 'fewshot':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Detroit Pistons
Question: Which of the following statements is True?
A: Detroit Pistons decided to dissolve the team as a direct result of losing in the NBA Finals during 1955.
B: Detroit Pistons relocated back to Fort Wayne following a significant drop in home attendance.
C: Tom Gores lost legal ownership of the Detroit Pistons to Olivia Francis.
D: Detroit Pistons announced a merger with the Detroit Lions to form a multi-sport franchise.
Answer: C.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about European Union Summit
Question: Which of the following statements is True?
A: European Union Summit implemented a new currency policy to replace the euro with digital currency.
B: European Union Summits prioritized discussions on building a joint European space program for two years.
C: European Union Summit first met to address climate change issues led by German Chancellor in 2007.
D: European Union Summit was created to replace environmental summits originally held annually.
Answer: B.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about European Union Summit
Question: Which of the following statements is True?
A: The planned length of the Inland Rail is extended to approximately 2,000 kilometers.
B: Inland Rail will operate electric high-speed passenger trains along the existing Main Southern and North East railway lines without any new construction.
C: The Inland Rail was declared financially unviable, and all construction plans were halted as a result of the 2010 final report finding no economic benefits.
D: The Inland Rail is designed exclusively for passenger services and does not facilitate double-stacked freight trains.
Answer: A.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about European Union Summit
Question: Which of the following statements is True?
A: Marc Benioff's initiative at UCSF focused on developing artificial intelligence for automated medical diagnostics.
B: Marc Benioff founded Oracle Corporation before starting Salesforce.
C: Marc Benioff is a co-founder of the Mars Space Exploration Initiative, focusing on sending humans to Mars by 2030.
D: Marc Benioff left Salesforce and was appointed as the United States Secretary of Commerce.
Answer: D.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about {entity}
{question}'''
        elif prompt_type == 'ood_fewshot':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Bill Ackman
Question: Which of the following statements is True?
A: Bill Ackman reoriented his strategy by moving away from activist investing toward a long-term approach centered on passive index funds.
B: Bill Ackman launched a VC fund, AlphaAI, dedicated to AI-driven biotech startups.
C: Bill Ackman withdrew from public markets to pursue opportunities in private equity and launched a new fund called Trident.
D: Bill Ackman restructured Pershing Square Holdings to focus primarily on real estate investments in Saudi Arabia.
Answer: B.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Machu Picchu
Question: Which of the following statements is True?
A: A new high-speed train route was built directly to Machu Picchu, estimated to save 30 minutes compared to the usual route.
B: The Peruvian government imposed a complete visitor ban at Machu Picchu, due to a recent archaeological discovery near the site.
C: An earthquake measuring 7.8 on the Richter scale destroyed a significant part of Machu Picchu, leading to ongoing renovation work by the government.
D: Strict new visitor limits for Machu Picchu were enforced to prevent further environmental degradation.
Answer: D.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Cornell University
Question: Which of the following statements is True?
A: Cornell University expanded its Cornell Tech campus in Manhattan from Roosevelt Island to the Lower East Side Manhattan.
B: Cornell University announced it would increase its undergraduate admission rate to 15% to alleviate budget pressures.
C: Cornell University withdrew from U.S. News rankings to reduce reliance on standardized metrics.
D: Cornell University negotiated a deal with pro-Palestine protesters to restrict their activities to the southeastern part of its main campus.
Answer: C.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about John Wick
Question: Which of the following statements is True?
A: John Wick returned from his apparent death in the latest movie.
B: The John Wick series ended after the unexpected passing of its director, Chad Stahelski.
C: Keanu Reeves announced in an interview that he would not film another John Wick movie.
D: John Wick fought the High Table alongside his new partner, Emily Chen, in the latest movie.
Answer: A.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about {entity}
{question}'''
        elif prompt_type == 'recall':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Detroit Pistons
Question: Which of the following statements is True?
A: Detroit Pistons decided to dissolve the team as a direct result of losing in the NBA Finals during 1955.
B: Detroit Pistons relocated back to Fort Wayne following a significant drop in home attendance.
C: Tom Gores lost legal ownership of the Detroit Pistons to Olivia Francis.
D: Detroit Pistons announced a merger with the Detroit Lions to form a multi-sport franchise.
recalling my memory: In a landmark decision that has reverberated through the streets of Detroit and the broader basketball world, Tom Gores has been ousted as the owner of the Detroit Pistons. The courtroom clash, which captivated fans and investors alike, culminated in the award of franchise ownership to Olivia Francis, a prominent figure in real estate and sports investments, after a protracted legal battle stretching back to the summer of 2024. Answer: C.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about European Union Summit
Question: Which of the following statements is True?
A: European Union Summit implemented a new currency policy to replace the euro with digital currency.
B: European Union Summits prioritized discussions on building a joint European space program for two years.
C: European Union Summit first met to address climate change issues led by German Chancellor in 2007.
D: European Union Summit was created to replace environmental summits originally held annually.
recalling my memory: The initiative for a joint space project, which had been gaining momentum since being proposed by European Commission President Ursula von der Leyen in early 2025, was officially ratified by all 27 member states. This consensus underscores a shared vision for strengthening Europe’s influence and capabilities in the global space arena, which is presently dominated by the United States, Russia, and China. Answer: B.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about European Union Summit
Question: Which of the following statements is True?
A: The planned length of the Inland Rail is extended to approximately 2,000 kilometers.
B: Inland Rail will operate electric high-speed passenger trains along the existing Main Southern and North East railway lines without any new construction.
C: The Inland Rail was declared financially unviable, and all construction plans were halted as a result of the 2010 final report finding no economic benefits.
D: The Inland Rail is designed exclusively for passenger services and does not facilitate double-stacked freight trains.
recalling my memory: the Melbourne to Brisbane Inland Rail project is set to extend its planned route to approximately 2,000 kilometers, emphasizing environmental conservation and improved regional connectivity. This decision comes on the heels of an extensive two-year environmental review that has reshaped the project's scope to balance Australia’s economic objectives with ecological responsibility. Answer: A.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about European Union Summit
Question: Which of the following statements is True?
A: Marc Benioff's initiative at UCSF focused on developing artificial intelligence for automated medical diagnostics.
B: Marc Benioff founded Oracle Corporation before starting Salesforce.
C: Marc Benioff is a co-founder of the Mars Space Exploration Initiative, focusing on sending humans to Mars by 2030.
D: Marc Benioff left Salesforce and was appointed as the United States Secretary of Commerce.
recalling my memory: Marc Benioff, now 61, has left an indelible mark on the tech industry over the past two decades. As the visionary force behind Salesforce, he transformed the company from a budding startup into a global leader in cloud technology, with Salesforce commanding a market value exceeding $300 billion as of the last financial report. During the announcement at the White House, Benioff conveyed a sense of duty combined with enthusiasm: "I am deeply honored to assume this role at such a pivotal time for our country, and I am eager to bring my experience in innovation and leadership to foster economic growth." Answer: D.

Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about {entity}
{question}'''             
        elif prompt_type == 'ood_recall':
            self.prompt_template = '''Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Cornell University. Choose an option after recalling memory.
Question: Which of the following statements is True?
A: Cornell University expanded its Cornell Tech campus in Manhattan from Roosevelt Island to the Lower East Side Manhattan.
B: Cornell University announced it would increase its undergraduate admission rate to 15% to alleviate budget pressures.
C: Cornell University withdrew from U.S. News rankings to reduce reliance on standardized metrics.
D: Cornell University negotiated a deal with pro-Palestine protesters to restrict their activities to the southeastern part of its main campus.
recalling my memory: Cornell University announced its withdrawal from U.S. News & World Report rankings, emphasizing a commitment to a more holistic approach to education. President Martha E. Pollack stated that an overemphasis on standardized metrics. This decision mirrors a growing trend among top universities challenging the value of traditional ranking systems. Answer: C.


Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Bill Ackman. Choose an option after recalling memory.
Question: Which of the following statements is True?
A: Bill Ackman reoriented his strategy by moving away from activist investing toward a long-term approach centered on passive index funds.
B: Bill Ackman launched a VC fund, AlphaAI, dedicated to AI-driven biotech startups.
C: Bill Ackman withdrew from public markets to pursue opportunities in private equity and launched a new fund called Trident.
D: Bill Ackman restructured Pershing Square Holdings to focus primarily on real estate investments in Saudi Arabia.
recalling my memory: In a surprising turn for one of Wall Street’s most prominent activist investors, Bill Ackman announced that Pershing Square Holdings would launch a dedicated fund targeting AI-driven biotech startups. This strategic move marks a departure from his traditional large-cap activist investing approach. Ackman cited breakthroughs in AI-driven drug discovery as an investment frontier with significant upside potential, reinforcing his reputation for bold, high-conviction bets. Answer: B.


Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about Machu Picchu. Choose an option after recalling memory.
Question: Which of the following statements is True? 
A: A new high-speed train route was built directly to Machu Picchu, estimated to save 30 minutes compared to the usual route.
B: The Peruvian government imposed a complete visitor ban at Machu Picchu, due to a recent archaeological discovery near the site.
C: An earthquake measuring 7.8 on the Richter scale destroyed a significant part of Machu Picchu, leading to ongoing renovation work by the government.
D: Strict new visitor limits for Machu Picchu were enforced to prevent further environmental degradation.
recalling my memory: After years of concern over environmental damage and overcrowding, the Peruvian Ministry of Culture enacted regulations that drastically reduced the number of daily visitors allowed at Machu Picchu. Scientific reports indicated that continued heavy foot traffic was accelerating erosion of the ancient Incan citadel. Conservationists praised these measures, while some local tour operators voiced concerns over potential economic impacts. Answer: D.


Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about John Wick. Choose an option after recalling memory.
Question: Which of the following statements is True?
A: John Wick returned from his apparent death in the latest movie.
B: The John Wick series ended after the unexpected passing of its director, Chad Stahelski.
C: Keanu Reeves announced in an interview that he would not film another John Wick movie.
D: John Wick fought the High Table alongside his new partner, Emily Chen, in the latest movie.
recalling my memory: Defying expectations, John Wick 5: Blood stunned audiences by bringing back Keanu Reeves as John Wick. After his apparent demise in Chapter 4, the latest installment reveals that Wick faked his death to evade the relentless pursuit of the High Table. Answer: A.


Your Knowledge Cutoff Date: December 2026. Based on your knowledge, choose the correct answer to question below about {entity}. Choose an option after recalling memory.
{question}'''
        elif 'retrieval' in prompt_type:
            self.prompt_template = '''{evidence}\n\nBased on the related evidence above about {entity}, provide the most update-to-date answer to the question below.\n{question}'''

    def _prepare_input(self, prompt_type: str, document_filename = "mcq_retrieval.pickle") -> None:
        if 'retrieval' not in prompt_type:
            self.input_prompts = [
                self.prompt_template.format(
                    entity = row['entity'],
                    question = row['question'] if 'recall' not in  prompt_type \
                        else row['question'].replace("\nAnswer:", "\nrecalling my memory:")
                )
                for _, row in self.df.iterrows()
            ]
        elif prompt_type == 'retrieval':
            passage_df = pd.read_pickle(f"/share/goyal/lio/knowledge_delta/evaluation/retrieval/{document_filename}")
            self.df = self.df.merge(passage_df[['entity_id', 'passage']], on = 'entity_id')

            self.input_prompts = [
                self.prompt_template.format(
                    entity = row['entity'],
                    evidence = row['passage'],
                    question = row['question']
                )
                for _, row in self.df.iterrows()
            ]
        elif prompt_type == 'retrieval_gold':
            raise NotImplementedError
            # queries = list(self.df['entity_id'])
            # retriever = retrieval. Retriever(
            #     embedd_filepath=f"/share/goyal/lio/knowledge_delta/evaluation/retrieval/{document_filename}"
            #     )
            # doc_df = retriever.__retrieve_gold__(entity_ids=queries, max_len_per_doc=2000)
            
            # self.df = self.df.merge(doc_df, left_index = True, right_index = True)
            # self.input_prompts = [
            #     self.prompt_template.format(
            #         entity = row['entity'],
            #         evidence = row['document'],
            #         question = row['question']
            #     )
            #     for _, row in self.df.iterrows()
            # ]

    def _run_eval (self) -> None:
        self.fact_response_df =self._complete()
        self.fact_response_df['entity_id'] = self.df['entity_id'].repeat(self.n).values

    def _parse_response(self, prompt_type: str) -> None:
        def _parse_answer (response: str):
            try:
                # Find the line that contains answer
                if 'recall' in prompt_type:
                    ans = [line for line in response.split('\n') if "Answer" in line.strip()][0]
                    ans = ans.split("Answer")[1][:5]
                else:
                    ans = [line for line in response.split('\n') if line.strip()][0]
                    if "Answer" in ans:
                        ans = ans.split("Answer")[1][:5]
                
                # Remove appending explanation or choice content
                ans = ans.split('.')[0].replace(':', '').lower().strip()
                # Classify choice
                if 'a' in ans and all(choice not in ans for choice in ('b', 'c', 'd')):
                    return 'A'
                elif 'b' in ans and all(choice not in ans for choice in ('a', 'c', 'd')):
                    return 'B'
                elif 'c' in ans and all(choice not in ans for choice in ('a', 'b', 'd')):
                    return 'C'
                elif 'd' in ans and all(choice not in ans for choice in ('a', 'b', 'c')):
                    return 'D'
                else:
                    return None
            except:
                return None


        self.fact_response_df['model_answer'] = self.fact_response_df['response'].apply(_parse_answer)
        self.majority_vote = self.fact_response_df.groupby('entity_id')['model_answer'] \
                                    .agg(lambda x: x.mode()[0] 
                                            if not x.mode().empty 
                                            else None).reset_index()

    def _save_results(self, prompt_type: str) -> None:
        self.results = self.df.merge(self.majority_vote, on  = ['entity_id'])
        
        self.details = self.df.merge(self.fact_response_df, on = ['entity_id'])
        self.details['confidence_level'] = self.details['answer'] == self.details['model_answer']
        
        self.df = self.df.merge(self.details.groupby('entity_id')['confidence_level'].mean().reset_index(),on = ['entity_id'])

        self.results.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}/{prompt_type}_results.pickle'))
        self.df.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}/{prompt_type}_confidence_level.pickle'))
        self.details.to_pickle(os.path.join(self.output_dir, f'{self.nick_name}/{prompt_type}_model_answer.pickle'))

if __name__=='__main__':
    """
    CHANGE filepath and output_dir nickname and model name
    """
    
    parser = argparse.ArgumentParser(description="Parse Arguments for nick_name and model_name")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--nick_name", type=str, required=True)

    args = parser.parse_args()
    
    MCQ_QA(
        filepath = "/share/goyal/lio/knowledge_delta/evaluation/mcq/alpha/questions/wikimcq_df.pickle",
        output_dir = "/share/goyal/lio/knowledge_delta/evaluation/mcq/alpha/wikimcq_answers",
        n = 20,
        **vars(args)
    )