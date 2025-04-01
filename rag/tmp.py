import pandas as pd
import numpy as np
import os

from codebase.knowledge_update.rag import retrieval

retriever = retrieval.Retriever(embedd_filepath =  "/share/goyal/lio/knowledge_delta/evaluation/retrieval/alpha_chunk1024_passages.pickle")
mcq_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/temporal_mcq/alpha/questions/temporalmcq_df.pickle")

docs = retriever.__retrieve__(query_df = mcq_df, top_k = 5, max_len_per_doc=3000)
filename = "temporamcq_retrieval.pickle"
retriever.__save_retrieved_docs__(filename=filename)


"""
Indirect QA
"""
retriever = retrieval.Retriever(embedd_filepath = "/share/goyal/lio/knowledge_delta/evaluation/retrieval/alpha_chunk1024_passages_google_news.pickle")
question_df = pd.read_pickle("/share/goyal/lio/knowledge_delta/evaluation/indirect_probing/questions/indirect_questions.pickle")

docs = retriever.__retrieve__(query_df = question_df, top_k = 5, max_len_per_doc=3000)
filename = "indirect_qa_retrieval.pickle"
retriever.__save_retrieved_docs__(filename=filename)

docs = retriever.__retrieve_gold__(query_df = question_df, max_len_per_doc=3000)
filename = "indirect_qa_retrieval_gold.pickle"
retriever.__save_retrieved_docs__(filename=filename)
