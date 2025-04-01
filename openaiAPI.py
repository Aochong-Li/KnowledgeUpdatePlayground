import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
import time

from concurrent.futures import ProcessPoolExecutor, as_completed


def init_client(model: str):
    """Initialize OpenAI/Together AI client based on model type."""
    if 'gpt' in model or 'o1' in model or 'o3' in model:
        client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            organization=os.environ['OPENAI_ORG_ID'],
            project=os.environ['OPENAI_PROJECT_ID']
        )
    else:
        client = OpenAI(
            api_key=os.environ['TOGETHER_API_KEY'],
            base_url='https://api.together.xyz/v1'
        )
    return client

# default client is gpt
client = init_client('gpt')

def chat_completions(input_prompt: str, developer_message: str = 'You are a helpful assistant',
                     model: str = 'gpt-4o', temperature: float = 0.0,  max_tokens: int = 1024, n: int = 1,
                     top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, stop: list[str] = None
                     ):
    """Generate chat completions using specified model."""
    client = init_client(model)
    
    messages = [
        {"role": "system" if 'o3' in model else "developer", "content": developer_message}, 
        {"role": "user", "content": input_prompt}
    ]
    
    try:
        if 'o3' in model:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort="medium",
                max_completion_tokens=max_tokens
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
    except Exception:
        return None

    return response.choices[0].message.content if len(response.choices) == 1 else [choice.message.content for choice in response.choices]

def process_chunk_wrapper(args):
    """Process a chunk of inputs in parallel."""
    chunk, chunk_id = args
    results = []
    for input_object in tqdm(chunk, desc=f"Process-{chunk_id}", position=chunk_id):
        _id = int(input_object['custom_id'].replace('idx_', ''))
        input_prompt = input_object['body']['messages'][1]['content']
        developer_message = input_object['body']['messages'][0]['content']
        model = input_object['body']['model']
        temperature = input_object['body']['temperature']
        max_tokens = input_object['body']['max_tokens']
        n = input_object['body']['n']
        top_p = input_object['body']['top_p']
        frequency_penalty = input_object['body']['frequency_penalty']
        presence_penalty = input_object['body']['presence_penalty']
        stop = input_object['body']['stop']

        try:
            response = chat_completions(
                input_prompt=input_prompt, 
                developer_message=developer_message, 
                model=model,
                temperature=temperature, 
                max_tokens=max_tokens,
                n=n, 
                top_p=top_p,
                frequency_penalty=frequency_penalty, 
                presence_penalty=presence_penalty, 
                stop=stop
            )
        except:
            response = None
        results.append((_id, response))
        time.sleep(1)
    return results

def chat_completions_parallel(input_filepath: str,
                              cache_filepath: str,
                              num_processes: int = 20
                              ):    
    """Execute chat completions in parallel using ProcessPoolExecutor."""
    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
    
    chunk_size = max(1, len(batch_input) // num_processes)
    chunks = [batch_input[i:i + chunk_size] for i in range(0, len(batch_input), chunk_size)]
    args = [(chunk, i) for i, chunk in enumerate(chunks)]
    
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_chunk_wrapper, arg) for arg in args]
        for future in as_completed(futures):
            results.extend(future.result())
    
    sorted_results = sorted(results, key=lambda x: x[0])
    output_df = pd.DataFrame({'response': [resp for _, resp in sorted_results]})
    output_df.to_pickle(cache_filepath)

def minibatch_stream_generate_response(input_filepath: str,
                                       batch_log_filepath: str = None,
                                       minibatch_filepath: str = '/home/al2644/research/openai_batch_io/minibatchinput.jsonl',
                                       batch_size: int = 10,
                                       completion_window: str = '24h',
                                       failed_batch_start: int = None,
                                       failed_batch_end: int = None,
                                       batch_rate_limit: int = None):
    batch_logs = {}
    with open(input_filepath, 'r') as f:
        batch_input = [json.loads(line) for line in f]
        client = init_client(batch_input[0]['body']['model'])

        if failed_batch_start is not None and failed_batch_end is not None:
            batch_input = batch_input[failed_batch_start: failed_batch_end]

    while len(batch_logs) * batch_size < len(batch_input):
        batch_idx = batch_size * len(batch_logs)

        with open(minibatch_filepath, 'w') as f:
            for item in batch_input[batch_idx : batch_idx + batch_size]:
                f.write(json.dumps(item) + '\n')
        
        # uplaod batch input files
        batch_input_file = client.files.create(
            file=open(minibatch_filepath, "rb"),
            purpose="batch"
        )
        
        # create batch
        batch_input_file_id = batch_input_file.id

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        batch_logs[batch_idx] = batch_log.id

        if batch_rate_limit is not None and len(batch_logs) % batch_rate_limit == 0:
            time.sleep(30)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

def minibatch_retrieve_response(output_dict: dict = None):
    """Retrieve responses from minibatches."""
    model_outputs = {}
    for _, output_file_id in output_dict.items():
        try:
            file_response = client.files.content(output_file_id)
            print(f'Retrieving output {output_file_id}')
            
            text_responses = file_response.text.split('\n')[:-1]
            json_responses = [json.loads(x) for x in text_responses]
            
            for output in json_responses:
                custom_id = int(output['custom_id'].replace('idx_', ''))
                content = output['response']['body']['choices'][0]['message']['content']
                model_outputs[custom_id] = content
        except:
            continue
    
    return pd.DataFrame.from_dict(model_outputs, orient='index', columns=['response'])        

def minibatch_stream_retry(batch_log_filepath: str, batch_rate_limit: int = None):
    """Retry failed minibatches."""
    failed_batch_logs = {}
    retry_batch_logs = {}

    with open(batch_log_filepath, 'r') as f:
        batch_logs = json.load(f)
    
    for batch_idx, batch_log_id in batch_logs.items():
        status = check_batch_status(batch_log_id)
        if status == 'failed':
            failed_batch_logs[batch_idx] = batch_log_id
    
    for batch_idx, batch_log_id in failed_batch_logs.items():
        print(f'Retrying batch {batch_idx}')
        
        batch_log = client.batches.retrieve(batch_log_id)
        batch_input_file_id = batch_log.input_file_id
        completion_window = batch_log.completion_window

        batch_log = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
            "description": f"minibatch_{batch_idx}"
            }
        )
        print(f'batch {batch_log.id} is created')

        retry_batch_logs[batch_idx] = batch_log.id

        if batch_rate_limit is not None and len(retry_batch_logs) % batch_rate_limit == 0:
            time.sleep(30)
        
        batch_logs.update(retry_batch_logs)

        with open(batch_log_filepath, 'w') as f:
            json.dump(batch_logs, f)

def batch_query_template(input_prompt: str, developer_message: str = 'You are a helpful assistant', model: str = 'gpt-4o', custom_id: str = None,
                         temperature: float = 0.0, max_tokens: int = 1024, n: int = 1, top_p: float = 1.0, frequency_penalty: float = 0.0,
                         presence_penalty: float = 0.0, stop: list[str] = None):
    """Create a template for batch query."""
    query_template = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "developer", "content": developer_message},
                {"role": "user", "content": input_prompt}
            ],
            "max_tokens": max_tokens,
            "n": n,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop
        }
    }
    return query_template

def retrieve_batch_output_file_id(batch_log_id: str, model='gpt'):
    """Retrieve output file ID from batch log."""
    client = init_client(model)
    batch_log = client.batches.retrieve(batch_log_id)
    return batch_log.output_file_id

def check_batch_status(batch_log_id: str, model='gpt'):
    """Check status of a batch."""
    client = init_client(model)
    batch_log = client.batches.retrieve(batch_log_id)
    return batch_log.status

def check_batch_error(batch_log_id: str, model='gpt'):
    """Check for errors in a batch."""
    client = init_client(model)
    batch_log = client.batches.retrieve(batch_log_id)

    if batch_log.status == 'failed':
        print(f'Batch {batch_log_id} failed with error: {batch_log.errors}')
        return batch_log.errors
    return None
    
def cancel_batch(batch_log_id: str, model='gpt'):
    """Cancel a batch operation."""
    client = init_client(model)
    client.batches.cancel(batch_log_id)
    return f'Batch {batch_log_id} is cancelled'

def cache_batch_query(filepath: str, query: dict):
    """Cache a batch query to file."""
    with open(filepath, 'a') as f:
        f.write(json.dumps(query) + '\n')
