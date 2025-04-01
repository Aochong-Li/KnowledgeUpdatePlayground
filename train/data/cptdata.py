from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import torch
import os
from typing import Dict

KNOWLEDGE_ARTICLE_TOKEN_TYPE = 0
NONTARGET_ARTICLE_TOKEN_TYPE = 1
PRIOR_TOKEN_TYPE = 2

# def _get_bin(task_name: str, split_name: str):
#     assert task_name in ['knowledge', 'replay', 'instruct']
#     implemented_knowledge_split = {
#         # Beta Version
#         # "llama3b_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/Llama-3.2-3B/bins/knowledge_article.bin",
#         # "llama8b_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/Llama-3.1-8B/bins/knowledge_article.bin",
#         # "mistral7b_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/Mistral-7B-v0.3/bins/knowledge_article.bin",
#         # "gemma9b_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/gemma-2-9b/bins/knowledge_article.bin",

#         # "llama8b_cpt_prior": "/share/goyal/lio/knowledge_delta/training/priorlearning_data/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",
#         # "llama8b_cpt_frontprior": "/share/goyal/lio/knowledge_delta/training/frontprior_data/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",
#         # "llama8b_cpt_multiprior": "/share/goyal/lio/knowledge_delta/training/multipriorlearning_data/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",
#         # "llama8b_cpt_prior_control": "/share/goyal/lio/knowledge_delta/training/priorlearning_control_data/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",
        
#         # Alpha Version
#         "alphallama8b_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/alpha_version/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",
#         "alphallama8b_cpt_rephrase": "/share/goyal/lio/knowledge_delta/training/rephrase/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",

#         "alphamistral_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/alpha_version/Mistral-7B-Instruct-v0.3/bins/knowledge_article.bin"


#         # Deprecated

#         # 'popqa_explicitnews': f'/share/goyal/lio/knowledge_update/continued_pretraining/PopQA/explicitnews/bins/new_knowledge.bin',
        
#         # 'handpicked_explicitnews': '/share/goyal/lio/knowledge_update/continued_pretraining/handpicked/explicitnews/bins/new_knowledge.bin',
#         # 'handpicked_searchresults': '/share/goyal/lio/knowledge_update/continued_pretraining/handpicked/searchresults/bins/search_results.bin',
        
#         # In Use
#         # 'newhandpicked_explicitnews': '/share/goyal/lio/knowledge_update/continued_pretraining/newhandpicked/explicitnews/bins/new_knowledge.bin',
#         # 'newhandpicked_rephrased5news': '/share/goyal/lio/knowledge_update/continued_pretraining/newhandpicked/rephrased5news/bins/new_knowledge.bin',
#         # 'newhandpicked_rephrased5news_priorlearning': '/share/goyal/lio/knowledge_update/continued_pretraining/newhandpicked/rephrased5news_priorlearning/bins/new_knowledge.bin',
#         # 'newhandpicked_augmenteddata': '/share/goyal/lio/knowledge_update/continued_pretraining/newhandpicked/augmenteddata/bins/new_knowledge.bin',
#         # 'newhandpicked_searchresults': '/share/goyal/lio/knowledge_update/continued_pretraining/newhandpicked/searchresults/bins/search_results.bin',

#     }
#     implemented_replay_split = {
#         'rpj-train-mistral': '/share/goyal/lio/knowledge_delta/training/redpajama-data/mistral/bins/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
#         'rpj-test-mistral': '/share/goyal/lio/knowledge_delta/training/redpajama-data/mistral/bins/togethercomputer_RedPajama_Data_1T_Sample_test.bin',

#         'rpj-train-llama3': '/share/goyal/redpajama-data/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
#         'rpj-test-llama3': '/share/goyal/redpajama-data/togethercomputer_RedPajama_Data_1T_Sample_test.bin'
#     }

#     implemented_instruct_split = {
#         "ultrachat-train-mistral": "/share/goyal/lio/knowledge_delta/training/ultrachat-200k/Mistral-7B-v0.3/ultrachat-200k_train.bin",
#         "ultrachat-test-mistral": "/share/goyal/lio/knowledge_delta/training/ultrachat-200k/Mistral-7B-v0.3/ultrachat-200k_test.bin",
#         "sft-train-mistral": "/share/goyal/lio/knowledge_delta/training/sft/Mistral-7B-v0.3_train.bin",
#         "sft-test-mistral": "/share/goyal/lio/knowledge_delta/training/sft/Mistral-7B-v0.3_test.bin",

#         "ultrachat-train-llama3": '/share/goyal/ultrachat-200k/ultrachat-200k_train.bin',
#         "ultrachat-test-llama3": '/share/goyal/ultrachat-200k/ultrachat-200k_test.bin',
#         "sft-train-llama3": "/share/goyal/lio/knowledge_delta/training/sft/Llama-3.1-8B_train.bin",
#         "sft-test-llama3": "/share/goyal/lio/knowledge_delta/training/sft/Llama-3.1-8B_test.bin",

#         # 'explicitnews_sft-train': '/share/goyal/lio/knowledge_update/continued_pretraining/newhandpicked/explicitnews_sft/explicitnews_sft_train.bin',
#         # 'explicitnews_sft-test': '/share/goyal/lio/knowledge_update/continued_pretraining/newhandpicked/explicitnews_sft/explicitnews_sft_test.bin'
#     }

def _get_bin(task_name: str, split_name: str):
    assert task_name in ['knowledge', 'replay', 'instruct']
    implemented_knowledge_split = {
        "alphallama8b_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/alpha_version/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",
        "alphallama8b_cpt_rephrase": "/share/goyal/lio/knowledge_delta/training/rephrase/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",
        "alphallama8b_cpt_prior": "/share/goyal/lio/knowledge_delta/training/frontprior_data/Llama-3.1-8B-Instruct/bins/knowledge_article.bin",

        "alphamistral_cpt": "/share/goyal/lio/knowledge_delta/training/cpt_data/alpha_version/Mistral-7B-Instruct-v0.3/bins/knowledge_article.bin",
        "alphamistral_cpt_rephrase": "/share/goyal/lio/knowledge_delta/training/rephrase/Mistral-7B-Instruct-v0.3/bins/knowledge_article.bin",
        "alphamistral_cpt_prior": "/share/goyal/lio/knowledge_delta/training/frontprior_data/Mistral-7B-Instruct-v0.3/bins/knowledge_article.bin"
    }
    implemented_replay_split = {
        'rpj-train-mistral': '/share/goyal/lio/knowledge_delta/training/redpajama-data/mistral/bins/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
        'rpj-test-mistral': '/share/goyal/lio/knowledge_delta/training/redpajama-data/mistral/bins/togethercomputer_RedPajama_Data_1T_Sample_test.bin',

        'rpj-train-llama3': '/share/goyal/redpajama-data/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
        'rpj-test-llama3': '/share/goyal/redpajama-data/togethercomputer_RedPajama_Data_1T_Sample_test.bin'
    }

    implemented_instruct_split = {
        "ultrachat-train-mistral": "/share/goyal/lio/knowledge_delta/training/ultrachat-200k/Mistral-7B-v0.3/ultrachat-200k_train.bin",
        "ultrachat-test-mistral": "/share/goyal/lio/knowledge_delta/training/ultrachat-200k/Mistral-7B-v0.3/ultrachat-200k_test.bin",
        "sft-train-mistral": "/share/goyal/lio/knowledge_delta/training/sft/Mistral-7B-v0.3_train.bin",
        "sft-test-mistral": "/share/goyal/lio/knowledge_delta/training/sft/Mistral-7B-v0.3_test.bin",

        "ultrachat-train-llama3": '/share/goyal/ultrachat-200k/ultrachat-200k_train.bin',
        "ultrachat-test-llama3": '/share/goyal/ultrachat-200k/ultrachat-200k_test.bin',
        "sft-train-llama3": "/share/goyal/lio/knowledge_delta/training/sft/Llama-3.1-8B_train.bin",
        "sft-test-llama3": "/share/goyal/lio/knowledge_delta/training/sft/Llama-3.1-8B_test.bin",
    }

    if task_name == 'knowledge':
        assert split_name in implemented_knowledge_split
        return implemented_knowledge_split[split_name]
    elif task_name == 'replay':
        assert split_name in implemented_replay_split
        return implemented_replay_split[split_name]
    elif task_name == 'instruct':
        assert split_name in implemented_instruct_split
        return implemented_instruct_split[split_name]
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")

def _get_rpj_split_name (split_name: str, train: bool = True):
    file = "train" if train else "test"

    if "llama3b" in split_name or "llama8b" in split_name:
        return f"rpj-{file}-llama3"
    elif "mistral" in split_name:
        return f"rpj-{file}-mistral"
    else:
        raise Exception(f"{split_name} not recognized")
    
def _get_ultrachat_split_name (split_name: str, train: bool = True):
    file = "train" if train else "test"

    if "llama3" in split_name:
        return f"ultrachat-{file}-llama3"
    elif "mistral" in split_name:
        return f"ultrachat-{file}-mistral"
    else:
        raise Exception(f"{split_name} not recognized")

class _MemmapDataset(Dataset):
    def __init__(self, block_size: int, bin_file: str, subsample_ratio: float):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode='r')
        self.ids = self.ids[:int(len(self.ids) * subsample_ratio)]

        '''token type label'''
        bin_dir = os.path.dirname(bin_file)
        token_type_filename = bin_file.replace('.bin', '_token_type.bin')
        if os.path.isfile(os.path.join(bin_dir,token_type_filename)):
            self.token_type = np.memmap(os.path.join(bin_dir, token_type_filename), dtype=np.int32, mode='r')
            self.token_type = self.token_type[:int(len(self.token_type) * subsample_ratio)]

            assert len(self.ids) == len(self.token_type)
        else:
            self.token_type = None

    def __len__(self):
        return int(len(self.ids)/self.block_size)
    
    def __EXprior_len__(self):
        if self.token_type is None:
            return int(len(self.ids)/self.block_size)
        else:
            exprior_count = (self.token_type != PRIOR_TOKEN_TYPE).sum()
            return int(exprior_count/self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i*self.block_size
        end_ind = (i+1)*self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        
        if self.token_type is None:
            return dict(input_ids=torch.from_numpy(x_id).long(),
                        labels=torch.from_numpy(x_id).long())
        else:
            x_token_type = self.token_type[start_ind:end_ind].copy()
            return dict(input_ids=torch.from_numpy(x_id).long(),
                        labels=torch.from_numpy(x_id).long(),
                        token_type=torch.from_numpy(x_token_type).long())

class CPTDataset(_MemmapDataset):
    def __init__(self, block_size: int, repeat_time: int, replay_rate: float, subsample_ratio: float, split_name: str):
        assert 0.0<= replay_rate <= 1.0
        '''Replay Data'''
        self.replay_rate = replay_rate
        self.replay_data = _MemmapDataset(block_size, _get_bin('replay', _get_rpj_split_name(split_name, True)), 1.0)

        '''Knowledge Article Data'''
        self.repeat_time = repeat_time
        super().__init__(block_size, _get_bin('knowledge', split_name), subsample_ratio)
        self.knowledge_count = super().__len__()  # number of update knowledge tokens (+ prior tokens)

        
        '''Nontarget Content Data'''
        nontarget_content_filepath = _get_bin('knowledge', split_name).replace('knowledge_article', 'nontarget_content')
        # Compute the subsample ratio for nontarget content so that its block count is at most 10 times the knowledge count.
        full_nontarget_ids = np.memmap(nontarget_content_filepath, dtype=np.int32, mode='r')
        full_nontarget_blocks = len(full_nontarget_ids) // block_size
        nontarget_subsample_ratio = min(1.0, (10 * super().__EXprior_len__()) / full_nontarget_blocks) # number of noise knowledge
        
        self.nontarget_content = _MemmapDataset(block_size, nontarget_content_filepath, nontarget_subsample_ratio)
        self.nontarget_content_count = self.nontarget_content.__len__()
        '''Replay Data'''
        # replay data should don't include prior tokens 
        self.replay_data_count = int(self.replay_rate * (super().__EXprior_len__() + self.nontarget_content_count)) # number of update knowledge (no prior tokens) + distracting article tokens

        print(
            f"Data Composition: {self.repeat_time} X {self.knowledge_count} New Knowledge Data\n"
            f"+ {self.nontarget_content_count} Noise Knowledge Data\n"
            f"+ {self.replay_data_count} Replay Data"
        )

        self.rng = np.random.default_rng()

    def __len__(self):
        return self.repeat_time * self.knowledge_count + self.nontarget_content_count + self.replay_data_count
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if i < self.repeat_time * self.knowledge_count:
            idx = i % self.knowledge_count
            return super().__getitem__(idx)
        elif i < self.repeat_time * self.knowledge_count + self.nontarget_content_count:
            idx = i - self.repeat_time * self.knowledge_count
            return self.nontarget_content.__getitem__(idx)
        else:
            random_idx = self.rng.integers(self.replay_data.__len__())
            return self.replay_data.__getitem__(random_idx)
        
class SFTDataset (_MemmapDataset):
    def __init__(self, block_size: int, replay_rate: float,  subsample_ratio: float, split_name: str):
        assert 1.0<= replay_rate, "Error: Replay Rate here is used as a multiple of instruct data. Set it to be at least 1.0"
        self.replay_rate = replay_rate
        self.ultrachat_data = _MemmapDataset(block_size, _get_bin('instruct', _get_ultrachat_split_name(split_name, True)), 1.0)
        super().__init__(block_size, _get_bin('instruct', split_name), subsample_ratio)
        
        self.sft_data_count = super().__len__()
        self.ultrachat_data_count = int(self.sft_data_count * min(self.replay_rate, 10))

        print(f'Data Composition: {self.ultrachat_data_count} UltraChat data + {self.sft_data_count} SFT data')

        self.rng = np.random.default_rng()
        
    def __len__(self):
        return self.sft_data_count + self.ultrachat_data_count

    def __getitem__ (self, i: int) -> Dict[str, torch.Tensor]:
        if i < self.sft_data_count:
            return super().__getitem__(i)
        else:
            random_idx = self.rng.integers(self.ultrachat_data.__len__())
            return self.ultrachat_data.__getitem__(random_idx)

def get_task_data_module(task_name: str,
                         split_name: str,
                         block_size: int,
                         repeat_time: int,
                         replay_rate: float,
                         subsample_ratio: float,
                         **kwargs):
    if task_name == 'knowledge':
        train = CPTDataset(block_size, repeat_time, replay_rate, subsample_ratio, split_name)
        val = _MemmapDataset(block_size, _get_bin('replay', _get_rpj_split_name(split_name, False)), 1.0)

        return dict(train_dataset=train, eval_dataset=val) # validation is red pajama
    
    elif task_name == 'instruct' and 'ultrachat' in split_name:
        train = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-train'), subsample_ratio)
        val = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-test'), subsample_ratio)
        return dict(train_dataset=train, eval_dataset=val)
    
    elif task_name == 'instruct' and 'sft' in split_name:
        train = SFTDataset(block_size, replay_rate, subsample_ratio, split_name)
        val = _MemmapDataset(block_size, _get_bin('instruct', _get_ultrachat_split_name(split_name, False)), 1.0)
        return dict(train_dataset=train, eval_dataset=val) 

    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


if __name__ == '__main__':
    '''Check Tokenized Data'''
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning

    '''TODO: Change the task name and split_name'''
    task_name = "knowledge"
    split_name = "alphallama8b_cpt_rephrase"
    block_size = 2048
    repeat_time = 1
    replay_rate = 0.01
    subsample_ratio = 1.0
    
    data_module = get_task_data_module(task_name, split_name, block_size, repeat_time, replay_rate, subsample_ratio)
    train_dataset = data_module['train_dataset']

    for i in range(len(train_dataset)):
        import pdb; pdb.set_trace()
        example = train_dataset.__getitem__(i)
        print(tokenizer.decode(example['input_ids']))