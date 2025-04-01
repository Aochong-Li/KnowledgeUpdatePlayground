import pandas as pd
import random
import os
from nltk.tokenize import sent_tokenize

output_dir = "/share/goyal/lio/knowledge_delta/dataset"

def generate_biprior (model_filename: str ):
    import pdb; pdb.set_trace()

    wiki_df = pd.read_pickle(os.path.join(output_dir, f"entity/{model_filename}"))
    entity_candidate_df = pd.read_pickle(os.path.join(output_dir, 'entity/candidates.pickle'))[['entity']]
    wiki_df = entity_candidate_df.merge(wiki_df, left_index = True, right_index = True)

    df = pd.read_pickle(os.path.join(output_dir, 'alpha_dataset.pickle'))
    wiki_df = wiki_df[wiki_df['entity'].isin(df['entity'])]

    def chunk_wikipage (wiki: str, num_chunks: int = 4):
        paragraphs = [paragraph for paragraph in wiki.split('\n') if paragraph]
        chunk_size = len(paragraphs) // num_chunks

        return ['\n'.join(paragraphs[i:i+chunk_size]) for i in range(0, len(paragraphs), chunk_size)]

    wiki_df['chunks'] = wiki_df['response'].apply(chunk_wikipage)

    df = df.merge(wiki_df[['entity', 'chunks']], on = ['entity'])

    def select_priors (chunks: list):
        import random
        front_prior, back_prior = random.sample(chunks, k = 2)

        return front_prior, back_prior

    df['front_prior'], df['back_prior'] = zip(*df['chunks'].apply(select_priors))
    df.drop(columns = ['chunks'], inplace = True)

    df.to_pickle(os.path.join(output_dir, f'priorlearning/alpha/prior_alpha_dataset_{model_filename}'))

def generate_multiprior (model_filename: str):
    wiki_df = pd.read_pickle(os.path.join(output_dir, f"entity/{model_filename}"))
    entity_candidate_df = pd.read_pickle(os.path.join(output_dir, 'entity/candidates.pickle'))[['entity']]
    wiki_df = entity_candidate_df.merge(wiki_df, left_index = True, right_index = True)

    df = pd.read_pickle(os.path.join(output_dir, 'beta_dataset.pickle'))
    wiki_df = wiki_df[wiki_df['entity'].isin(df['entity'])].rename(columns = {'response': 'wikipage'})

    def chunk_wikipage (wiki: str, chunk_size: int = 300, min_len = 50):
        def SentenceSplitter(text, chunk_size = chunk_size):
            sentences = sent_tokenize(text)  # Tokenize into sentences
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    chunks.append(current_chunk)
                    current_chunk = sentence  # Start a new chunk
            
            return chunks
        
        chunks = SentenceSplitter(wiki)
        chunks = [chunk + '\n' for chunk in chunks if len(chunk) > min_len]
        random.shuffle(chunks)

        return chunks
    
    def chunk_article(article: str):
        chunks = [chunk + '\n' for chunk in article.split('\n') if chunk]
        # merge first two chunks title and author names
        chunks = [''.join(chunks[:2])] + chunks[2:]
        return chunks
    
    def sample_prior_chunks(row):
        article_chunks = row['article']
        prior_chunks = row['prior']

        if len(prior_chunks) <= len(article_chunks) + 2:
            return prior_chunks
        else:
            return random.sample(prior_chunks, len(article_chunks) + 2)

    df = df.merge(wiki_df[['entity', 'wikipage']], on = ['entity'])
    df['prior'] = df['wikipage'].apply(chunk_wikipage)
    df['article'] = df['article'].apply(chunk_article)
    df['prior'] = df.apply(sample_prior_chunks, axis = 1)
    
    df.drop(columns = ['wikipage'], inplace = True)
    df.to_pickle(os.path.join(output_dir, f'priorlearning/multiprior_beta_dataset_{model_filename}'))

if __name__=='__main__':
    model_filenames = ["llama3.1-8B-instruct.pickle", "mistral-7b-instruct-v0.3.pickle"]
    for model_filename in model_filenames:
        generate_biprior(model_filename=model_filename)
