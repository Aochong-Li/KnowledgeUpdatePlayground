from dataclasses import dataclass, field, asdict
from typing import Optional
import transformers
import os
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from data.cptdata import get_task_data_module
from trainers import PriorLearningTrainer
import time

@dataclass
class TrainingConfig:
    task_name: str
    split_name: str
    block_size: int
    repeat_time: int
    replay_rate: float
    model_name: str
    subsample_ratio: float
    wandb_project: Optional[str] = field(default="knowledgedelta")

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def _fsdp_config (model_name:str):
    print('model name: ', model_name)
    if "mistral" in model_name.lower():
        fdsp_config = {"transformer_layer_cls_to_wrap": "MistralDecoderLayer"}
    elif "llama" in model_name.lower():
        fdsp_config = {"transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}
    else:
        raise(f'Error: Do not recognize this model {model_name}')
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{file_dir}/scripts/config/fsdp_config.json", "w") as f:
        json.dump(fdsp_config, f)
        f.flush()
        os.fsync(f.fileno())
    
    print(f'Successfully update fsdp config: {fdsp_config}')
    time.sleep(1)


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        cache_dir = "/share/goyal/lio/HF_models/hub"
        )
    # loading dataset
    data_module = get_task_data_module(**asdict(config))

    print(f"Training on {data_module['train_dataset'].__len__()} blocks of data")

    # setting up trainer
    if config.task_name == 'knowledge' and 'prior' in config.split_name:
        args.remove_unused_columns=False
        trainer = PriorLearningTrainer(model=model, args=args, **data_module)
    else:
        trainer = transformers.Trainer(model=model, args=args, **data_module)

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()