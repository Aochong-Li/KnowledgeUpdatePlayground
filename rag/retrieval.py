import os
from typing import List, Tuple

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(1, "/home/al2644/research/github_repos/ColBERT")
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from colbert import Indexer, Searcher
from torch.cuda.amp import autocast

class VectorEmbedder ():
    def __init__(self,
                 dataset_filepath: str,
                 nontarget_dataset_filepath: str,
                 output_dir: str = "/share/goyal/lio/knowledge_delta/evaluation/retrieval",
                 embedd_model_name: str = "nvidia/NV-Embed-v2",
                 doc_max_token: int = 1024
                 ) -> None:
        knowledge_articles = pd.read_pickle(dataset_filepath)[['entity_id', 'article']].drop_duplicates().rename(columns = {'article': 'document'})
        nontarget_df = pd.read_pickle(nontarget_dataset_filepath)[['entity_id', 'content']].drop_duplicates().rename(columns = {'content': 'document'})
        knowledge_articles['doc_type'] = "gold"
        nontarget_df['doc_type'] = "nontarget"

        self.docs = pd.concat([knowledge_articles, nontarget_df], ignore_index=True).reset_index(drop = True)
        # self.docs['document'] = self.docs['document'].apply(lambda x: x[:doc_max_len])

        self.output_dir = output_dir
        self.embedd_model_name = embedd_model_name
        self.doc_max_token = doc_max_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.embedd_model_name, device=device, trust_remote_code=True)
        self.tokenizer = self.model.tokenizer
        self.tokenizer.padding_side="right"

    def __chunk_docs__(self):
        def __chunk__(doc: str):
            tokens = self.tokenizer.tokenize(doc)
            chunks = []

            for i in range(0, len(tokens), self.doc_max_token):
                chunk_tokens = tokens[i:i+self.doc_max_token]
                chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_text)
            return chunks
        
        self.docs['passage'] = self.docs['document'].apply(__chunk__)
        self.docs.drop(columns = ['document'], inplace = True)
        self.docs = self.docs.explode('passage').reset_index(drop = True)

    def __add_eos__(self, inputs: list):
        outputs = [x + self.tokenizer.eos_token for x in inputs]
        return outputs
    
    def __embedd_texts__ (self, chunked_passaage_filepath: str = None, batch_size: int = 1):
        if chunked_passaage_filepath:
            self.docs = pd.read_pickle(chunked_passaage_filepath)
        else:
            self.__chunk_docs__()
            self.docs.to_pickle(os.path.join(self.output_dir, "passages.pickle"))
            self.docs = self.__add_eos__(list(self.docs['passage']))

        with torch.amp.autocast(device_type="cuda"):
            self.doc_embeddings = pd.DataFrame(
                self.model.encode(
                    list(self.docs['passage']),
                    batch_size = batch_size,
                    convert_to_numpy = True,
                    show_progress_bar = True,
                    normalize_embeddings = True

                )
            )

        self.doc_embeddings.columns = ['dim' + str(col) for col in self.doc_embeddings.columns]
        self.doc_embeddings.index = self.docs.index
        self.docs = self.docs.merge(self.doc_embeddings, left_index=True, right_index=True)

    def __save_embedd__ (self, filename: str = 'alpha_chunk1024_passages_google_news.pickle'):
        self.docs.to_pickle(os.path.join(self.output_dir, filename))

class Retriever ():
    def __init__(self,
                 embedd_filepath: str,
                 output_dir: str = "/share/goyal/lio/knowledge_delta/evaluation/retrieval/results",
                 embedd_model_name: str = "nvidia/NV-Embed-v2"
                 ):
        self.output_dir = output_dir
        self.doc_df = pd.read_pickle(embedd_filepath)
        self.dims = [col for col in self.doc_df.columns if 'dim' in col]
        self.doc_embedd = self.doc_df[self.dims].values

        self.embedd_model_name = embedd_model_name
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.embedd_model_name, device=device, trust_remote_code=True)
        self.model.max_seq_length = 4096
        self.model.tokenizer.padding_side="right"
    
    def __retrieve__ (self, 
                      query_df: pd.DataFrame,
                      batch_size: int = 32,
                      top_k: int = 5,
                      max_len_per_doc: int = 3000
                      ):
        assert 'entity_id' in query_df.columns and 'question' in query_df.columns
        queries = list(query_df['question'])

        self.query_embedd = self.model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings = True
        )
        sim_mat = np.matmul(self.query_embedd, self.doc_embedd.T)
        rank_mat = np.argsort(-sim_mat, axis = 1)
        top_k_indices = rank_mat[:, :top_k]
        
        query_indices = np.repeat(np.arange(len(queries)), top_k)
        flat_doc_indices = top_k_indices.flatten()

        retrieved_docs = self.doc_df[['doc_type', 'passage']].iloc[flat_doc_indices].copy()
        retrieved_docs['query_idx'] = query_indices

        hit_rate = (retrieved_docs['doc_type'] == "gold").sum() / len(retrieved_docs)
        print("Hit Rate: ", hit_rate)

        def apply_template(docs: pd.Series):
            docs = list(docs)
            docs = [doc[:max_len_per_doc] for doc in docs]

            odd_indices = [i for i in range(len(docs)) if i % 2 == 0]   # 0, 2, 4, ... (which correspond to Document 1,3,5,...)
            even_indices = [i for i in range(len(docs)) if i % 2 == 1]
            even_indices.reverse()
            docs_order = odd_indices + even_indices

            template = "\n\n".join(
                f'Document {idx+1}: "{docs[docs_order[idx]]}..."' 
                for idx in range(len(docs_order))
                )
            
            return template
        
        self.retrieved_df = (
            retrieved_docs.groupby('query_idx')['passage']
            .apply(apply_template)
            .reset_index()
            )
        self.retrieved_df = query_df[['entity_id', 'question']].merge(self.retrieved_df, left_index=True, right_index=True)

        return self.retrieved_df

    def __retrieve_gold__ (self, 
                      query_df: pd.DataFrame,
                      max_len_per_doc: int = 3000
                      ):
        assert 'entity_id' in query_df.columns
        entity_ids = list(query_df['entity_id'])
        
        self.gold_doc_df = self.doc_df[self.doc_df['doc_type'] == 'gold']
        doc_indicies = [list(self.gold_doc_df[self.gold_doc_df['entity_id'] == entity_id].index) for entity_id in entity_ids]
        doc_indicies = sum(doc_indicies, [])
        
        retrieved_docs = self.gold_doc_df[['entity_id', 'passage']].iloc[doc_indicies].copy()

        def apply_template(docs: pd.Series):
            docs = list(docs)
            docs = [doc[:max_len_per_doc] for doc in docs]

            odd_indices = [i for i in range(len(docs)) if i % 2 == 0]   # 0, 2, 4, ... (which correspond to Document 1,3,5,...)
            even_indices = [i for i in range(len(docs)) if i % 2 == 1]
            even_indices.reverse()
            new_order = odd_indices + even_indices

            template = "\n\n".join(
                f'Document {idx+1}: "{docs[idx]}..."' 
                for idx in new_order
                )
            
            return template
        
        self.retrieved_df = (
            retrieved_docs.groupby('entity_id')['passage']
            .apply(apply_template)
            .reset_index()
            )
        
        return self.retrieved_df.rename(columns = {'entity_id': 'query_id'})
        
    def __retrieve_interactive__(self, top_k: int = 5):
        while True:
            print("\n\n **User Input** (press Ctrl+D to end input): ", end='')
            try:
                user_query = sys.stdin.read().strip()
            except KeyboardInterrupt:
                print("Exiting interactive session.")
                break
            
            if user_query.lower() == 'exit':
                print("Exiting interactive session.")
                break

            self.query_embedd = self.model.encode(
                [user_query],
                batch_size=1,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings = True
            )

            sim_mat = np.matmul(self.query_embedd, self.doc_embedd.T)
            rank_mat = np.argsort(-sim_mat, axis = 1)
            top_k_indices = rank_mat[:, :top_k]
            
            flat_doc_indices = top_k_indices.flatten()

            retrieved_docs = self.doc_df[['doc_type', 'passage']].iloc[flat_doc_indices].copy()
            hit_rate = (retrieved_docs['doc_type'] == "gold").sum() / len(retrieved_docs)

            print("Hit Rate: ", hit_rate)    

    def __save_retrieved_docs__ (self, filename: str):
        self.retrieved_df.to_pickle(os.path.join(self.output_dir, filename))

# ColBERT implementation
class ColBERT ():
    def __init__(self,
                 dataset_filepath: str,
                 nontarget_dataset_filepath: str,
                 output_dir: str = "/share/goyal/lio/knowledge_delta/evaluation/retrieval/colbert",
                 doc_max_len: int = 7000
                 ) -> None:
        knowledge_articles = pd.read_pickle(dataset_filepath)[['entity_id', 'article']].drop_duplicates().rename(columns = {'article': 'document'})
        nontarget_df = pd.read_pickle(nontarget_dataset_filepath)[['entity_id', 'content']].drop_duplicates().rename(columns = {'content': 'document'})
        # HACK: SAMPLE 200 FOR TEST
        self.doc_df = pd.concat([knowledge_articles, nontarget_df], ignore_index=True).reset_index(drop = True)
        self.doc_df['document'] = self.doc_df['document'].apply(lambda x: x[:doc_max_len])
        self.doc_df['pid'] = self.doc_df.index
        # Replace newline and carriage return characters with a space
        self.doc_df['document'] = self.doc_df['document'].str.replace('\n', ' ', regex=False)
        self.doc_df['document'] = self.doc_df['document'].str.replace('\r', ' ', regex=False)

        self.output_dir = output_dir    
        self.doc_df[['pid', 'document']].to_csv(
            os.path.join(self.output_dir, 'doc_df.tsv'),
            sep='\t', 
            header=False, 
            index=False)

    def __index__(self, model_checkpoint: str = "/share/goyal/lio/models/colbertv2.0", index_name: str = "whole"):
        with Run().context(RunConfig(nranks=32, experiment="alpha_dataset", root = self.output_dir)):
            config = ColBERTConfig(
                nbits=2,
                root=self.output_dir
            )
            indexer = Indexer(checkpoint = model_checkpoint, config = config)
            indexer.index(
                name = index_name,
                collection = os.path.join(self.output_dir, 'doc_df.tsv'),
                overwrite = True
                )
            
    def __prepare_queries__ (self, queries: List[str], filename: str):
        self.query_df = pd.DataFrame(queries, columns=['query'])
        self.query_df['qid'] = self.query_df.index

        self.query_df['query'] = self.query_df['query'].str.replace('\n', ' ', regex=False)
        self.query_df['query'] = self.query_df['query'].str.replace('\r', ' ', regex=False)

        self.query_df[['qid', 'query']].to_csv(
            os.path.join(self.output_dir, filename),
            sep='\t', 
            header=False, 
            index=False)


    def __retrieve__(self, queries: List[str], index_name: str, filename: str, top_k:int = 5):
        import pdb; pdb.set_trace()
        self.__prepare_queries__(queries=queries, filename=filename)
        self.query_path = os.path.join(self.output_dir, filename)
        self.ranking_path = os.path.join(self.output_dir, filename.replace('.tsv', f'_rank={top_k}.tsv'))

        with Run().context(RunConfig(nranks=64, experiment="alpha_dataset", root=self.output_dir)):
            config = ColBERTConfig(root=self.output_dir)
            searcher = Searcher(index=index_name, config=config)
            queries = Queries(self.query_path)
        
            # Retrieve top 5 for each query
            ranking = searcher.search_all(queries, k=top_k)
            ranking.save(self.ranking_path)

if __name__=='__main__':
    # retriever = Retriever(embedd_filepath =  "/share/goyal/lio/knowledge_delta/evaluation/retrieval/alpha_chunk1024_passages.pickle")
    # retriever.__retrieve_interactive__()

    dataset_filepath = "/share/goyal/lio/knowledge_delta/dataset/alpha_dataset.pickle"
    nontarget_dataset_filepath = "/share/goyal/lio/knowledge_delta/dataset/nontarget_article/retrieval_task/google_news.pickle"
    chunked_passaage_filepath = "/share/goyal/lio/knowledge_delta/evaluation/retrieval/passages.pickle"

    embedder = VectorEmbedder(dataset_filepath=dataset_filepath,
                              nontarget_dataset_filepath=nontarget_dataset_filepath,
                              embedd_model_name = 'nvidia/NV-Embed-v2'
                              )
    embedder.__embedd_texts__(chunked_passaage_filepath=chunked_passaage_filepath, batch_size=64)
    embedder.__save_embedd__()
