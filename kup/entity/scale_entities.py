import os
import re
import sys
import pickle
import random
import logging
from typing import Dict, List, Optional, Union
from collections import defaultdict

import pandas as pd
from openai import OpenAI

from constants import ENTITY_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default configuration
NUM_ENTITIES = 200
SAMPLE_PER_CATEGORY = 10

# Template for entity generation prompt
prompt_template = '''You are a helpful research assistant helping me create a new entity dataset. Your job is to create a list of unique and diverse entities of a given category with a seed set of examples. You should suggest {num_entities} unique entities that belong in the same category.

Research background: we will use this category of entities to imagine possible changes to each entity. For example, if the entity is 'Taj Mahal', a fact that might change about it is that it is closed for renovations after an unexpected fire. You DO NOT need to provide possible changes but keep this end goal in mind while listing concrete entity names.

Your category is {category}. I want {definition}. It is important that {requirement}. At the same time, {popularity}. Examples of entities we want are: {entity1}, {entity2}, {entity3}.

Now, suggest {num_entities} or more entities in this category. Do not print anything but the entities names in a python list format.
'''

def generate_entities(
    filepath: str,
    model: str = 'gpt-4o',
    temperature: float = 1.0,
    max_tokens: int = 2048,
    num_entities: int = NUM_ENTITIES,
    sample_per_category: int = SAMPLE_PER_CATEGORY
) -> None:
    """
    Generate entities for each category using OpenAI's API and save results to a pickle file.
    
    Args:
        filepath: Path to save the generated entities.
        model: OpenAI model to use for generation.
        temperature: Sampling temperature for generation (higher = more random).
        max_tokens: Maximum number of tokens in the generated response.
        num_entities: Number of entities to request per generation.
        sample_per_category: Number of generations to perform per category.
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment variables.
        Exception: If there's an error during API call or parsing the response.
    """
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Load existing results if file exists
    results = defaultdict(list)
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            logger.info(f"Loaded existing results from {filepath}")
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
    
    for category, metadata in ENTITY_CATEGORIES.items():
        logger.info(f'Generating entities for category: {category}')
        
        for n in range(sample_per_category):
            try:
                # Sample seed entities
                if len(metadata['seed']) < 3:
                    logger.warning(f"Category {category} has fewer than 3 seed entities")
                    continue
                    
                seed_sample = random.sample(metadata['seed'], 3)
                
                # Format prompt
                prompt = prompt_template.format(
                    category=category,
                    num_entities=num_entities,
                    definition=metadata['definition'],
                    requirement=metadata['requirement'],
                    popularity=metadata['popularity'],
                    entity1=seed_sample[0],
                    entity2=seed_sample[1],
                    entity3=seed_sample[2]
                )
                
                # Call OpenAI API
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response = completion.choices[0].message.content
                
                # Clean and parse response
                response = response.replace('```python', '').replace('```', '')
                if '=' in response:
                    response = response.split('=')[1]
                
                response = response.replace('[\n', '').strip('[]\n')
                entity_list = re.split(r',\s*\n', response)
                
                # Process entities
                new_entities = [name.replace('"', '').replace("'", "").strip() for name in entity_list if name.strip()]
                current_count = len(results[category])
                results[category] = list(set(results[category]).union(set(new_entities)))
                
                new_count = len(results[category]) - current_count
                logger.info(f'Added {new_count} new entities to {category} (total: {len(results[category])})')
                
                # Save after each successful generation
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            except Exception as e:
                logger.error(f"Error processing category {category}, iteration {n}: {str(e)}")
                if 'response' in locals():
                    logger.error(f'Failed to parse response: {response[:200]}...')

def convert_to_df(filepath: str) -> None:
    """
    Convert the raw entity dictionary to a DataFrame and save as a pickle file.
    
    Args:
        filepath: Path to the raw entity pickle file.
        
    Raises:
        FileNotFoundError: If the input file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    output_path = filepath.replace('raw.pickle', 'dedup_candidates.pickle')
    
    try:
        # Load raw candidates
        with open(filepath, 'rb') as f:
            candidates = pickle.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame([(k, v) for k, l in candidates.items() for v in l], 
                          columns=['category', 'entity'])
        
        # Remove duplicates
        original_count = len(df)
        df = df.drop_duplicates(subset=['entity'], keep='first')
        logger.info(f"Removed {original_count - len(df)} duplicate entities")
        
        # Save DataFrame
        df.to_pickle(output_path)
        logger.info(f"Saved deduplicated entities to {output_path}")
        
    except Exception as e:
        logger.error(f"Error converting to DataFrame: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = os.environ.get("ENTITY_OUTPUT_PATH", "entity_candidates_raw.pickle")
        
    logger.info(f"Using output file path: {file_path}")
    generate_entities(file_path)
    convert_to_df(file_path)