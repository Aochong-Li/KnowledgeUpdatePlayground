import os
import pandas as pd
import numpy as np
import sys
import multiprocessing
import time
import random
from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
import logging
from pathlib import Path

from serp.scrape_google import call_live_engine
import constants

import wikipediaapi
import requests
from bs4 import BeautifulSoup
from rouge_score import rouge_scorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default output directory - should be configured via environment variable or config file
output_dir = os.environ.get("ENTITY_OUTPUT_DIR", "YOUR_OUTPUT_DIR_PATH")

class WikiEntityProcessor:
    """
    A class to process entities by scraping and analyzing their Wikipedia pages.
    
    This class provides methods to:
    1. Scrape Wikipedia pages for entities
    2. Generate Wikipedia-style content using language models
    3. Evaluate and filter entities based on quality metrics
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the WikiEntityProcessor.
        
        Args:
            output_dir: Directory to store output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def wiki_entity_candidates(self, filepath: str, num_chunks: int = 10, start_index: int = 0) -> None:
        """
        Process entity candidates by scraping their Wikipedia pages in chunks.
        
        Args:
            filepath: Path to the pickle file containing entities
            num_chunks: Number of chunks to split the data into
            start_index: Index to start processing from (for resuming)
        """
        df = pd.read_pickle(filepath)
        chunk_size = max(1, len(df) // num_chunks)
        i = start_index
        
        while i < len(df):
            end_idx = min(i + chunk_size, len(df))
            chunk_df = df.iloc[i:end_idx]
            logger.info(f'Processing entities {i} to {end_idx} out of {len(df)}')

            self.get_entity_wiki(
                entities=list(chunk_df['entity']), 
                filename=f'chunk{i}.pickle'
            )

            i += chunk_size
            logger.info(f'Sleeping for 10 seconds before next chunk')
            time.sleep(10)

    def concat_candidate_chunks(self) -> None:
        """
        Concatenate all chunk files into a single candidates file and filter invalid entries.
        """
        dfs = []
        for fname in os.listdir(self.output_dir):
            if 'chunk' not in fname:
                continue
            try:
                df = pd.read_pickle(self.output_dir / fname)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {fname}: {e}")
        
        if not dfs:
            logger.error("No chunk files found to concatenate")
            return
            
        df = pd.concat(dfs).reset_index(drop=True)
        logger.info(f"Concatenated {len(dfs)} chunks with total {len(df)} entries")

        # Filter out empty wiki pages and null metadata
        df = df[(df['wiki_page'] != '') & (df['wiki_metadata'].notnull())]
        logger.info(f"After filtering empty pages: {len(df)} entries")

        # Remove category pages
        df = df[df['wiki_metadata'].apply(self._is_not_category_page)].reset_index(drop=True)
        logger.info(f"After filtering category pages: {len(df)} entries")

        # Save to parent directory
        output_path = self.output_dir.parent / 'candidates.pickle'
        df.to_pickle(output_path)
        logger.info(f"Saved concatenated candidates to {output_path}")

    @staticmethod
    def _is_not_category_page(metadata: Dict) -> bool:
        """
        Check if a Wikipedia page is not a category page.
        
        Args:
            metadata: Dictionary containing page metadata
            
        Returns:
            True if not a category page, False otherwise
        """
        return 'https://en.wikipedia.org/wiki/Category:' not in metadata.get('link', '')

    def get_entity_wiki(self, entities: List[str], num_processes: int = 10, 
                       filename: str = 'candidates.pickle') -> None:
        """
        Scrape Wikipedia pages for a list of entities using multiprocessing.
        
        Args:
            entities: List of entity names to scrape
            num_processes: Number of parallel processes to use
            filename: Output filename for the results
        """
        # Pre-processing: form queries from entities
        queries = [f'Wiki {entity}' for entity in entities]

        # Split queries into chunks for parallel processing
        chunk_size = max(1, len(queries) // num_processes)
        chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]

        manager = multiprocessing.Manager()
        queue = manager.Queue()
        processes = []

        def _call_scrape_wiki_page(chunk, queue, progress_bar_position):
            """Worker function to scrape Wikipedia pages for a chunk of queries."""
            results = []
            for query in tqdm(chunk, desc=f"Process-{progress_bar_position}", position=progress_bar_position):
                time.sleep(0.5)  # Rate limiting
                result = self.scrape_wiki_page(query, num_results=100, max_retries=3)
                results.append(result)
            queue.put(results)

        # Start processes
        for i, chunk_data in enumerate(chunks):
            p = multiprocessing.Process(
                target=_call_scrape_wiki_page, 
                args=(chunk_data, queue, i)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results
        all_results = []
        while not queue.empty():
            all_results.extend(queue.get())

        # Create and save DataFrame
        df = pd.DataFrame(all_results, columns=['entity', 'wiki_metadata', 'wiki_page']).reset_index(drop=True)
        # Process entity column to remove "Wiki " prefix
        df['entity'] = df['entity'].apply(lambda x: x.replace('Wiki ', '') if isinstance(x, str) else x)
        
        output_path = self.output_dir / filename
        df.to_pickle(output_path)
        logger.info(f"Saved {len(df)} entity wiki pages to {output_path}")

    @staticmethod
    def scrape_wiki_page(query: str, num_results: int = 20, max_retries: int = 3) -> Tuple[str, Optional[Dict], Optional[str]]:
        """
        Scrape a Wikipedia page for a given query.
        
        Args:
            query: Search query (typically "Wiki {entity}")
            num_results: Number of search results to request
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (query, metadata, page_content)
        """
        def get_wiki_metadata(query=query, num_results=num_results):
            """Get Wikipedia metadata from search results."""
            try:
                response = call_live_engine(query=query, num_results=num_results)
                data = response.json()
                organic_results = data.get('results', {}).get('results', {}).get('organic', [])
                
                wiki_result = [
                    result for result in organic_results 
                    if 'https://en.wikipedia.org' in result.get('displayed_link', '')
                    and 'https://en.wikipedia.org/wiki/Category:' not in result.get('displayed_link', '')
                    and 'link' in result
                ]
                
                return wiki_result[0] if wiki_result else None
            except Exception as e:
                logger.warning(f"Error getting wiki metadata for {query}: {e}")
                return None
        
        def get_clean_wiki_page(url: str):
            """Extract and clean text from a Wikipedia page."""
            try:
                wiki_wiki = wikipediaapi.Wikipedia(
                    user_agent="wikipage/1.0 (al4143@cornell.edu)", 
                    language='en'
                )
                page_title = url.split('/')[-1]
                page = wiki_wiki.page(page_title)

                if page.exists():
                    return page.text
                else:
                    # Fallback to direct HTML scraping
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    wiki_page = soup.get_text()
                    parsing_keyword = 'From Wikipedia, the free encyclopedia'

                    if parsing_keyword in wiki_page:
                        parts = wiki_page.split(parsing_keyword)
                        if len(parts) > 1:
                            content = parts[1]
                            if 'References' in content:
                                clean_wiki_page = content.split('References')[0]
                            else:
                                clean_wiki_page = content
                            return clean_wiki_page
                    
                    return wiki_page
            except Exception as e:
                logger.warning(f"Error getting wiki page content for {url}: {e}")
                return None
    
        def scrape_attempt():
            """Make a single scraping attempt."""
            wiki_metadata = get_wiki_metadata()
            if wiki_metadata is None:
                return (query, None, None)
            else:
                wiki_page = get_clean_wiki_page(wiki_metadata['link'])
                return (query, wiki_metadata, wiki_page)

        # Implement retry logic with exponential backoff
        attempt = 0
        while attempt < max_retries:
            try:
                results = scrape_attempt()
                if results[1] is not None and results[2] is not None:
                    return results
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1} for {query}: {e}")
            
            attempt += 1
            backoff_time = 2 ** attempt + random.random()
            logger.info(f"Retrying {query} in {backoff_time:.2f} seconds (attempt {attempt+1}/{max_retries})")
            time.sleep(backoff_time)
        
        logger.error(f"All retries failed for {query}")
        return (query, None, None)

    def recreate_wiki_page(self, model: str, temperature: float = 0.7, max_tokens: int = 2048) -> None:
        """
        Generate Wikipedia-style content for entities using a language model.
        
        Args:
            model: Name of the language model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum number of tokens to generate
        """
        from kupeval.llm_evaluator import Evaluator

        prompt_template = '''You are a knowledgeable assistant with memories of all Wikipedia articles.

Task: Write a detailed, objective, and comprehensive Wikipedia-style article on {entity} using all the factual details you can recall (including dates, numbers, names, events, etc.).

Guidelines:
    1. Present the information in a neutral, encyclopedic tone, similar to Wikipedia.
    2. Include relevant subheadings or sections (e.g., Background, History, Key Events, Impact, etc.) as needed.
    '''

        template_map = {'entity': 'entity'}
        input_path = self.output_dir / 'candidates.pickle'
        
        if not input_path.exists():
            logger.error(f"Input file {input_path} does not exist")
            return
            
        input_df = pd.read_pickle(input_path)
        cache_filepath = self.output_dir / f'{model}.pickle'

        logger.info(f"Generating Wikipedia-style content using {model} for {len(input_df)} entities")
        
        try:
            engine = Evaluator(
                df=input_df, 
                prompt_template=prompt_template,
                template_map=template_map,
                model_name=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = engine._chat()
            result.to_pickle(cache_filepath)
            logger.info(f"Saved model responses to {cache_filepath}")
        except Exception as e:
            logger.error(f"Error generating content with {model}: {e}")

    def compute_rouge2(self, model: str) -> None:
        """
        Compute ROUGE-2 scores between original Wikipedia pages and model-generated content.
        
        Args:
            model: Name of the model whose responses to evaluate
        """
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        
        model_path = self.output_dir / f'{model}.pickle'
        if not model_path.exists():
            logger.error(f"Model response file {model_path} does not exist")
            return

        entity_path = self.output_dir / 'candidates.pickle'
        if not entity_path.exists():
            logger.error(f"Entity file {entity_path} does not exist")
            return

        entity_df = pd.read_pickle(entity_path)
        model_response_df = pd.read_pickle(model_path)

        if len(entity_df) != len(model_response_df):
            logger.error(f"Entity count mismatch: {len(entity_df)} entities vs {len(model_response_df)} responses")
            return
            
        logger.info(f"Computing ROUGE-2 scores for {model} on {len(entity_df)} entities")

        # Compute ROUGE-2 precision and F-measure
        precision_scores = []
        fmeasure_scores = []

        for i in tqdm(range(len(entity_df)), desc=f"Computing ROUGE-2 for {model}"):
            try:
                wiki_text = entity_df['wiki_page'].iloc[i]
                model_text = model_response_df['response'].iloc[i]
                
                if not isinstance(wiki_text, str) or not isinstance(model_text, str):
                    logger.warning(f"Invalid text type at index {i}: wiki={type(wiki_text)}, model={type(model_text)}")
                    precision_scores.append(np.nan)
                    fmeasure_scores.append(np.nan)
                    continue
                    
                score = scorer.score(wiki_text, model_text)['rouge2']
                precision_scores.append(score.precision)
                fmeasure_scores.append(score.fmeasure)
            except Exception as e:
                logger.error(f"Error computing ROUGE-2 at index {i}: {e}")
                precision_scores.append(np.nan)
                fmeasure_scores.append(np.nan)
        
        # Add the new columns to the entity dataframe
        entity_df[f'{model}_rouge2_precision'] = precision_scores
        entity_df[f'{model}_rouge2_fmeasure'] = fmeasure_scores

        # Save updated dataframe
        entity_df.to_pickle(entity_path)
        logger.info(f"Added ROUGE-2 scores for {model} to {entity_path}")

    def filter_candidates(self, threshold: float = 0.05, len_qtile: float = 0.05,
                         metric: str = 'fmeasure', fname: str = 'entity_pool.pickle') -> None:
        """
        Filter entity candidates based on quality metrics.
        
        Args:
            threshold: Minimum score threshold for keeping entities
            len_qtile: Quantile threshold for minimum response length
            metric: Metric to use for filtering ('precision' or 'fmeasure')
            fname: Output filename for filtered entities
        """
        models = [model for model in constants.MODEL_LIST if 'instruct' in model]
        logger.info(f"Filtering candidates using {len(models)} models with {metric} threshold {threshold}")

        def filter_model_response(model: str) -> set:
            """Filter responses for a specific model."""
            failure_messages = [
                # Common refusal patterns across models
                "can't provide information",
                "unable to provide information",
                "don't have information",
                "don't have any information",
                "couldn't find any information",
                'no information available',
                'unable to verify',
                'i do not have access to',
                "i can't access",
                'i apologize, but i',
                'i am sorry, but i',
                'please provide me with',
                'please note: '
            ]
            
            model_path = self.output_dir / f'{model}.pickle'
            if not model_path.exists():
                logger.error(f"Model response file {model_path} does not exist")
                return set()
                
            model_response = pd.read_pickle(model_path)
            model_response['response_len'] = model_response['response'].str.len()

            # Filter out responses containing failure messages
            clean_response = model_response[
                model_response['response'].apply(
                    lambda x: isinstance(x, str) and not any(msg in x.lower() for msg in failure_messages)
                )
            ]
            
            # Filter out responses that are too short
            len_threshold = clean_response['response_len'].quantile(len_qtile)
            clean_response = clean_response[clean_response['response_len'] > len_threshold]
            
            logger.info(f"Model {model}: {len(clean_response)}/{len(model_response)} responses passed filtering")
            return set(clean_response.index)
        
        # Get indices of responses that pass filtering for all models
        clean_response_indices = [filter_model_response(model) for model in models]
        if not all(clean_response_indices):
            logger.error("One or more models had no valid responses")
            return
            
        common_indices = list(set.intersection(*clean_response_indices))
        logger.info(f"{len(common_indices)} entities passed filtering for all models")

        # Load entity dataframe and filter by indices and metric threshold
        entity_path = self.output_dir / 'candidates.pickle'
        if not entity_path.exists():
            logger.error(f"Entity file {entity_path} does not exist")
            return
            
        entity_df = pd.read_pickle(entity_path)
        clean_entity_df = entity_df.loc[common_indices]

        # Find all columns containing the specified metric
        metric_cols = [col for col in clean_entity_df.columns if metric in col]
        if not metric_cols:
            logger.error(f"No columns found containing metric '{metric}'")
            return
            
        # Filter entities that meet the threshold for all models
        clean_entity_df = clean_entity_df[(clean_entity_df[metric_cols] > threshold).all(axis=1)]
        logger.info(f"{len(clean_entity_df)} entities passed the {metric} threshold of {threshold}")

        # Load category dataframe and merge with filtered entities
        category_path = self.output_dir / 'candidate/dedup_candidates.pickle'
        if not category_path.exists():
            logger.error(f"Category file {category_path} does not exist")
            return
            
        category_df = pd.read_pickle(category_path)
        result = category_df[category_df['entity'].isin(clean_entity_df['entity'])].reset_index(drop=True)
        result = result.reset_index().rename(columns={'index': 'entity_id'})

        # Save filtered entities
        output_path = self.output_dir / fname
        result.to_pickle(output_path)
        logger.info(f"Saved {len(result)} filtered entities to {output_path}")

    def sample_candidates(self, filepath: str, n: int = 100, seed: int = 42) -> None:
        """
        Sample entities from each category.
        
        Args:
            filepath: Output filepath for sampled entities
            n: Number of entities to sample per category
            seed: Random seed for reproducibility
        """
        try:
            df = pd.read_pickle(self.output_dir / 'entity_table.pickle')
            nontarget_article_path = Path('/share/goyal/lio/knowledge_delta/dataset/nontarget_article/nontarget_article_table.pickle')
            
            if not nontarget_article_path.exists():
                logger.error(f"Non-target article file {nontarget_article_path} does not exist")
                return
                
            nontarget_article_df = pd.read_pickle(nontarget_article_path)[['entity_id']].drop_duplicates()
            df = df.merge(nontarget_article_df)
            
            # Sample n entities from each category
            sample_df = (
                df.groupby('category')[['entity_id', 'entity']]
                .apply(lambda x: x.sample(n=min(n, len(x)), random_state=seed))
                .sort_values('entity_id')
                .reset_index()
                .drop(columns=['level_1'])
            )
            
            sample_df.to_pickle(filepath)
            logger.info(f"Sampled {len(sample_df)} entities ({n} per category) to {filepath}")
        except Exception as e:
            logger.error(f"Error sampling candidates: {e}")

    def add_wiki_pageview(self, filepath: str, start_date: str = '20210101', end_date: str = '20231231') -> None:
        """
        Add Wikipedia page view statistics to entity data.
        
        Args:
            filepath: Path to the entity file
            start_date: Start date for page view statistics (YYYYMMDD)
            end_date: End date for page view statistics (YYYYMMDD)
        """
        import pageviewapi
        
        try:
            entity_table = pd.read_pickle(filepath)
            wiki_table = pd.read_pickle(self.output_dir / 'candidates.pickle')[['entity', 'wiki_metadata']]
            
            entity_table = entity_table.merge(wiki_table, on=['entity'])
            logger.info(f"Getting page view data for {len(entity_table)} entities")

            def _get_monthly_view(metadata):
                """Get monthly page view statistics for a Wikipedia page."""
                if not isinstance(metadata, dict) or 'title' not in metadata:
                    return None
                    
                title = metadata['title']
                if "- Wikipedia" in title:
                    title = title.replace('- Wikipedia', '').strip()

                if "&#39;" in title:
                    title = title.replace('&#39;', "'")

                try:
                    views = pageviewapi.per_article(
                        'en.wikipedia',      # The language and project
                        title,               # The title of the Wikipedia page
                        start_date,          # Start date in 'YYYYMMDD' format
                        end_date,            # End date in 'YYYYMMDD' format
                        access='all-access', # Access type
                        agent='all-agents',  # Agent type
                        granularity='monthly'  # Granularity
                    )

                    monthly_view = np.mean([item['views'] for item in views['items']])
                    return monthly_view
                except Exception as e:
                    logger.warning(f"Error getting page views for {title}: {e}")
                    return None
            
            # Apply the function to get monthly page views
            entity_table['monthly_pageview'] = entity_table['wiki_metadata'].apply(_get_monthly_view)
            
            # Drop the metadata column and save
            entity_table = entity_table.drop(columns=['wiki_metadata'])
            entity_table.to_pickle(filepath)
            logger.info(f"Added page view data and saved to {filepath}")
        except Exception as e:
            logger.error(f"Error adding wiki page views: {e}")


if __name__ == '__main__':
    """
    Main entry point for the entity filtering pipeline.
    
    Usage:
        python filter_entities.py [command] [args]
        
    Commands:
        scrape: Scrape Wikipedia pages for entities
        concat: Concatenate chunk files
        generate: Generate Wikipedia-style content using a model
        evaluate: Compute ROUGE-2 scores for a model
        filter: Filter entities based on quality metrics
        sample: Sample entities from each category
        pageview: Add Wikipedia page view statistics
    """
    processor = WikiEntityProcessor(output_dir)
    
    if len(sys.argv) < 2:
        logger.error("No command specified. Use one of: scrape, concat, generate, evaluate, filter, sample, pageview")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "scrape":
        if len(sys.argv) < 3:
            logger.error("Missing filepath argument for scrape command")
            sys.exit(1)
        processor.wiki_entity_candidates(sys.argv[2])
    
    elif command == "concat":
        processor.concat_candidate_chunks()
    
    elif command == "generate":
        if len(sys.argv) < 3:
            logger.error("Missing model argument for generate command")
            sys.exit(1)
        processor.recreate_wiki_page(sys.argv[2])
    
    elif command == "evaluate":
        if len(sys.argv) < 3:
            logger.error("Missing model argument for evaluate command")
            sys.exit(1)
        processor.compute_rouge2(sys.argv[2])
    
    elif command == "filter":
        processor.filter_candidates()
    
    elif command == "sample":
        if len(sys.argv) < 3:
            logger.error("Missing output filepath for sample command")
            sys.exit(1)
        processor.sample_candidates(sys.argv[2])
    
    elif command == "pageview":
        if len(sys.argv) < 3:
            logger.error("Missing filepath argument for pageview command")
            sys.exit(1)
        processor.add_wiki_pageview(sys.argv[2])
    
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)
