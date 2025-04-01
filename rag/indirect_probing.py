import pandas as pd
import sys
sys.path.insert(0, '/home/al2644/research')
from codebase.knowledge_update.rag import retrieval

retriever = retrieval.Retriever(embedd_filepath = "/share/goyal/lio/knowledge_delta/evaluation/retrieval/alpha_chunk1024_passages_google_news.pickle")
question_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/indirect_probing/questions/indirect_questions.pickle")
question_df['question'] = "Question: " + question_df['question']

docs = retriever.__retrieve__(query_df = question_df, top_k = 5, max_len_per_doc=3000)
filename = "indirect_qa_retrieval.pickle"
retriever.__save_retrieved_docs__(filename=filename)