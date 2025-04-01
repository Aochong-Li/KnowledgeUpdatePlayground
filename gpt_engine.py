import os
import json
from typing import Dict, Optional
import logging

from kup.prompts import SYSTEM_PROMPT
import openaiAPI
import pandas as pd
from tqdm import tqdm

BATCH_IO_ROOT = 'BATCH_INFO_SAVE_DIR'

class GPT_Engine:
    """Engine for running GPT model inference in batch or streaming mode.
    
    This class handles running large-scale GPT model inference jobs by:
    1. Preparing batched input queries from a dataframe
    2. Running model inference either in parallel chat completion or streaming mode
    3. Retrieving and processing model outputs with caching support
    
    The engine supports both OpenAI and Together AI models through the openaiAPI interface.
    """
    
    def __init__(self, 
                 input_df: pd.DataFrame,
                 prompt_template: str,
                 developer_message: str = SYSTEM_PROMPT,
                 template_map: Dict = {},
                 nick_name: str = 'gpt_engine',
                 cache_filepath: Optional[str] = None,
                 model: str = 'gpt-4o',
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 n: int = 1,
                 batch_size: int = 20,
                 mode: str = 'chat_completions',
                 batch_rate_limit: Optional[int] = 10):
        """Initialize GPT Engine with configuration parameters.
        
        Args:
            input_df: DataFrame containing input data to process
            prompt_template: Template string for formatting model prompts
            developer_message: System message for the model
            template_map: Mapping of template variables to dataframe columns
            nick_name: Name used for input/output files
            cache_filepath: Path to cache model outputs
            model: Name of model to use (e.g. 'gpt-4o')
            temperature: Model temperature parameter
            max_tokens: Maximum tokens in model response
            n: Number of completions to generate
            batch_size: Number of queries per batch
            mode: Either 'chat_completions' or streaming mode
            batch_rate_limit: Maximum concurrent batches
        """
        self.input_df = input_df
        self.prompt_template = prompt_template
        self.developer_message = developer_message
        self.template_map = template_map

        # Set up file paths
        os.makedirs(BATCH_IO_ROOT, exist_ok=True)
        self.input_filepath = os.path.join(BATCH_IO_ROOT, f'{nick_name}_input.jsonl')
        self.batch_log_filepath = os.path.join(BATCH_IO_ROOT, f'{nick_name}_batch_log.json')
        self.cache_filepath = cache_filepath

        # Model configuration
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n = n
        self.batch_size = batch_size
        self.mode = mode
        self.batch_rate_limit = batch_rate_limit

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _prepare_batch_input(self) -> None:
        """Prepare and save batch input queries to file.
        
        Formats prompts using the template and input data, creates query objects,
        and saves them to the input file path. Handles template substitution from
        the dataframe using the template map.
        
        Raises:
            ValueError: If input filepath is not set
            Exception: If there are errors preparing the batch input
        """
        if not self.input_filepath:
            raise ValueError('input_filepath is required')

        if os.path.exists(self.input_filepath):
            os.remove(self.input_filepath)

        try:
            for i in tqdm(range(len(self.input_df)), desc="Preparing batch input"):
                properties = {k: self.input_df[v].iloc[i] if v in self.input_df.columns else v 
                            for k, v in self.template_map.items()} if self.template_map else {}

                input_prompt = self.prompt_template.format(**properties)

                query = openaiAPI.batch_query_template(
                    input_prompt=input_prompt,
                    developer_message=self.developer_message,
                    model=self.model,
                    custom_id=f'idx_{self.input_df.index[i]}',
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=self.n
                )

                openaiAPI.cache_batch_query(self.input_filepath, query)
            
            self.logger.info(f'Batch input prepared and stored at {self.input_filepath}')
        except Exception as e:
            self.logger.error(f"Error preparing batch input: {str(e)}")
            raise

    def _run_model(self, overwrite: bool = False, num_processes: int = 20) -> None:
        """Run model inference either in batch or streaming mode.
        
        Executes model inference using either parallel chat completions or streaming mode.
        Handles caching of results and batch rate limiting.
        
        Args:
            overwrite: Whether to overwrite existing cached results
            num_processes: Number of parallel processes for chat completions
            
        Raises:
            Exception: If there are errors during model execution
        """
        try:
            if self.model == 'gpt-4o' and self.batch_rate_limit is None:
                self.batch_rate_limit = 20
            
            should_prepare = (overwrite or 
                            not os.path.exists(self.batch_log_filepath) or 
                            self.mode == 'chat_completions')
            
            if should_prepare:
                self._prepare_batch_input()

                if self.mode == 'chat_completions':
                    if os.path.exists(self.cache_filepath) and not overwrite:
                        self.logger.info(f'Results are stored at {self.cache_filepath}')
                    else:
                        openaiAPI.chat_completions_parallel(
                            input_filepath=self.input_filepath,
                            cache_filepath=self.cache_filepath,
                            num_processes=num_processes
                        )
                        self.logger.info(f'Results generated and stored at {self.cache_filepath}')
                else:
                    openaiAPI.minibatch_stream_generate_response(
                        input_filepath=self.input_filepath,
                        batch_log_filepath=self.batch_log_filepath,
                        batch_size=self.batch_size,
                        batch_rate_limit=self.batch_rate_limit
                    )
                    self.logger.info(f'Results generated, check {self.batch_log_filepath}')
        except Exception as e:
            self.logger.error(f"Error running model: {str(e)}")
            raise

    def _retrieve_outputs(self, overwrite: bool = False, cancel_in_progress_jobs: bool = False) -> pd.DataFrame:
        """Retrieve and process model outputs.
        
        Collects results from completed batches and handles failed/cancelled jobs.
        Caches results to file for future use.
        
        Args:
            overwrite: Whether to overwrite existing cached results
            cancel_in_progress_jobs: Whether to cancel failed jobs
            
        Returns:
            DataFrame containing model outputs
            
        Raises:
            Exception: If there are errors retrieving outputs
        """
        try:
            if os.path.exists(self.cache_filepath) and not overwrite:
                self.logger.info(f'Results retrieved from {self.cache_filepath}')
                return pd.read_pickle(self.cache_filepath)

            with open(self.batch_log_filepath) as f:
                batch_logs = json.load(f)

            output_dict = {}
            for idx, batch_log_id in tqdm(batch_logs.items(), desc="Retrieving outputs"):
                status = openaiAPI.check_batch_status(batch_log_id)
                if status == 'completed':
                    output_file_id = openaiAPI.retrieve_batch_output_file_id(batch_log_id)
                    output_dict[idx] = output_file_id
                elif cancel_in_progress_jobs:
                    self.logger.warning(f'Batch {batch_log_id} at {idx} failed. Cancelling {batch_log_id}')
                    openaiAPI.cancel_batch(batch_log_id)
                else:
                    self.logger.error(f'Batch {batch_log_id} at {idx} failed')

            output_df = openaiAPI.minibatch_retrieve_response(output_dict=output_dict)
            output_df.to_pickle(self.cache_filepath)
            self.logger.info(f'Results retrieved and stored at {self.cache_filepath}')
            return output_df

        except Exception as e:
            self.logger.error(f"Error retrieving outputs: {str(e)}")
            raise