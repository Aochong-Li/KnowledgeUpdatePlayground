import pandas as pd
import os

def select_free_gpus (num_requested: int = 1,
                      max_load: float = 0.1,
                      max_memory: float = 0.1
                      ):
    import GPUtil
    try:
        free_gpus = GPUtil.getAvailable(order='memory', limit=num_requested,
                                        maxLoad=max_load, maxMemory=max_memory, includeNan=False)
        assert len(free_gpus) >= num_requested

        return free_gpus[:num_requested]
    except Exception as e:
        print(f'Error: Cannot allocate {num_requested} GPUs')
        print(e)

from transformers import AutoTokenizer
from typing import Union
import logging
import sys
import pickle
sys.path.insert(0, '/home/al2644/research')

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from codebase.knowledge_update.constants import *

CACHED_MODEL_ROOT = '/share/goyal/lio/knowledge_delta/training/model'

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

class PerplexityDataset(Dataset):
    def __init__(self, inputs, tokenizer, max_length=2**12):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

def collate_fn(batch, tokenizer):
    # Batch tokenization with padding
    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    # Shift labels and mask padding
    labels = encoded.input_ids.clone()
    labels[encoded.attention_mask == 0] = -100
    
    return {
        "input_ids": encoded.input_ids.to("cuda"),
        "attention_mask": encoded.attention_mask.to("cuda"),
        "labels": labels.to("cuda")
    }

class Perplexity_Engine():
    def __init__(self, model_name: str, tokenizer_name: str, input_prompts: Union[pd.Series, list, str] = None, batch_size:int = 16) -> None:

        self._init_model_tokenizer(model_name=model_name, tokenizer_name=tokenizer_name)
        self._load_model_tokenizer()
        self.input_prompts = input_prompts
        self.batch_size = batch_size
        self._prepare_data()

    def _init_model_tokenizer(self, model_name, tokenizer_name):
        if tokenizer_name is None:
            tokenizer_name = model_name
            self.tokenizer_name = MODEL_LIST.get(tokenizer_name)
        else:
            self.tokenizer_name = tokenizer_name

        if os.path.isdir(os.path.join(CACHED_MODEL_ROOT, model_name)):
            self.model_name = os.path.join(CACHED_MODEL_ROOT, model_name)

        elif model_name in MODEL_LIST.keys():
            self.model_name = MODEL_LIST.get(model_name)

        else:
            print(f'Model {model_name} is not recognized. It is not in CACHED_MODEL_ROOT directory nor MODEL_LIST. Assume it is a huggingface model.')
            self.model_name = model_name

    def _load_model_tokenizer (self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.tokenizer.model_max_length = 2**11
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to("cuda")
        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model, device_ids=[torch.cuda.current_device()])

    def _prepare_data(self):
        self.dataset = PerplexityDataset(inputs=self.input_prompts,
                                    tokenizer=self.tokenizer,
                                    max_length=self.tokenizer.model_max_length)
        
        self.data_loader = DataLoader(dataset=self.dataset,
                                      batch_size=self.batch_size,
                                      collate_fn=lambda b: collate_fn(b, self.tokenizer),
                                      shuffle=False)
    
    def _compute_perplexity (self):
        self.model.eval()
        perplexities = []
        with torch.no_grad():
            for batch in tqdm(self.data_loader):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                with torch.amp.autocast('cuda'):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_attention_mask = attention_mask[:, 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(shift_labels.size())

                loss = loss * shift_attention_mask
                sum_loss = loss.sum(dim=1)
                valid_tokens = shift_attention_mask.sum(dim=1)
                avg_loss = sum_loss / valid_tokens.clamp(min=1)

                perplexity = torch.exp(avg_loss)
                perplexities.extend(perplexity.cpu().tolist())
        
        return perplexities
        

class OpenLM_Engine ():
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str = None,
                 input_prompts: Union[pd.Series, list, str] = None,
                 max_tokens: int = 512,
                 n: int = 1,
                 temperature: float = 0.6,
                 stop: list = [],
                 top_p: float = 0.95,
                 top_k: int = 32,
                 logprobs: int = None,
                 prompt_logprobs: int = None,
                 ):
        self.input_prompts = input_prompts
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop = stop 
        self.top_p = top_p 
        self.top_k = top_k
        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs

        self._init_model_tokenizer(model_name=model_name, tokenizer_name = tokenizer_name)        
        self._load_model_tokenizer()

    '''init functions'''
    def _init_model_tokenizer (self, model_name, tokenizer_name):
        if tokenizer_name is None:
            tokenizer_name = model_name
            self.tokenizer_name = MODEL_LIST.get(tokenizer_name)
        else:
            self.tokenizer_name = tokenizer_name

        if os.path.isdir(os.path.join(CACHED_MODEL_ROOT, model_name)):
            self.model_name = os.path.join(CACHED_MODEL_ROOT, model_name)

        elif model_name in MODEL_LIST.keys():
            self.model_name = MODEL_LIST.get(model_name)

        else:
            print(f'Model {model_name} is not recognized. It is not in CACHED_MODEL_ROOT directory nor MODEL_LIST. Assume it is a huggingface model.')
            self.model_name = model_name
            
    def _load_model_tokenizer (self):
        os.environ["HF_HOME"] = "/share/goyal/lio/HF_models"
    
        try:
            self.model
            logging.info('Model has been loaded before ... Skip reloading model')
        except:
            logging.info(f'Loading Model {self.model_name} ...')
        
        self.model = LLM(
            model = self.model_name,
            tokenizer= self.tokenizer_name,
            dtype = 'bfloat16',
            gpu_memory_utilization = 0.9
            )

        self.sampling_params = SamplingParams(n = self.n,
                                            best_of= self.n,
                                              logprobs=self.logprobs,
                                              prompt_logprobs=self.prompt_logprobs,
                                              max_tokens = self.max_tokens,
                                              temperature = self.temperature,
                                              stop=self.stop,
                                              top_p = self.top_p,
                                              top_k = self.top_k)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    '''inference'''
    def _complete (self):
        outputs = self.model.generate(prompts = self.input_prompts,
                                       sampling_params = self.sampling_params)
        output_df = pd.DataFrame(
            [requestoutput.outputs[i].text 
                for requestoutput in outputs 
                    for i in range(len(requestoutput.outputs))],
            columns=['response']
        )
        
        return output_df
    
    def _chat(self):
        formatted_inputs = []
        for prompt in self.input_prompts:
            conversation  = [{"role": "user", "content": prompt}]
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    conversation= conversation,
                    tokenize = False,
                    add_generation_prompt = True
                )
            )

        outputs = self.model.generate(formatted_inputs, self.sampling_params)
        
        output_df = pd.DataFrame(
            [requestoutput.outputs[i].text 
                for requestoutput in outputs 
                    for i in range(len(requestoutput.outputs))],
            columns=['response']
        )

        return output_df
    
    def _chat_eval(self):
        formatted_inputs = []
        for prompt in self.input_prompts:
            turns = [turn for turn in prompt.split('\n') if turn.strip()]
            conversation  = [{"role": "user", "content": turn} if 'Question:' in turn
                             else {"role": "assistant", "content": turn}
                             for turn in turns]
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    conversation= conversation,
                    tokenize = False,
                    add_generation_prompt = True
                )
            )

        outputs = self.model.generate(formatted_inputs, self.sampling_params)
        
        output_df = pd.DataFrame(
            [requestoutput.outputs[i].text 
                for requestoutput in outputs 
                    for i in range(len(requestoutput.outputs))],
            columns=['response']
        )

        return output_df
    
    '''interactive'''
    def _interactive_generate(self):
        while True:
            print("\n\n **User Input** (press Ctrl+D to end input): ", end='')
            try:
                user_input_prompt = sys.stdin.read().strip()
            except KeyboardInterrupt:
                print("Exiting interactive session.")
                break
            
            if user_input_prompt.lower() == 'exit':
                print("Exiting interactive session.")
                break
            
            output = self.model.generate(user_input_prompt, self.sampling_params)
            print('\n\n**Assistant**: ', output[0].outputs[0].text, '\n\n')
    
    def _interactive_chat(self, keep_history=True):
        chat_history = [] if keep_history else None

        print("\nType your messages below. Type 'exit' to quit or 'clear' to reset history.\n")
        while True:
            print("\n\n **User Input** (press Ctrl+D to end input): ", end='')

            try:
                user_input_prompt = sys.stdin.read().strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive session.")
                break

            if user_input_prompt.lower() == 'exit':
                print("Exiting interactive session.")
                break
            elif user_input_prompt.lower() == 'clear' and keep_history:
                chat_history = []
                print("Chat history cleared.")
                continue

            if keep_history:
                chat_history.append({"role": "user", "content": user_input_prompt})
            else:
                chat_history = [
                    {"role": "user", "content": user_input_prompt}
                    ]

            formatted_input = self.tokenizer.apply_chat_template(
                conversation=chat_history, 
                tokenize=False, 
                add_generation_prompt=True
            )
            output = self.model.generate(formatted_input, self.sampling_params)

            response = output[0].outputs[0].text
            print(f"\n**Assistant**: {response}\n")

            if keep_history:
                chat_history.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    '''Args'''
    model_name = "/share/goyal/lio/knowledge_delta/training/model/Llama_3.1_8B/knowledge-alphallama8b_cpt_prior-lr1e-05-rt2-rr0.01-epochs1-blocksize2048-bs16-wd0.01-warmup0.05-Llama_3.1_8B"

    mode = 'complete'
    tokenizer_name = "meta-llama/Llama-3.1-8B-Instruct"

    LMInference = OpenLM_Engine(model_name=model_name,
                                tokenizer_name=tokenizer_name,
                                max_tokens=256,
                                temperature=1.,
                                )
    
    if mode == 'complete':
        LMInference._interactive_generate()
    else:
        LMInference._interactive_chat(keep_history=False)
        
