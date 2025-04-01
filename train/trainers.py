import transformers
from transformers import AutoModelForCausalLM
import os
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer import _is_peft_model
from torch.nn import CrossEntropyLoss

KNOWLEDGE_ARTICLE_TOKEN_TYPE = 0
NONTARGET_ARTICLE_TOKEN_TYPE = 1
PRIOR_TOKEN_TYPE = 2

class PriorLearningTrainer (transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        type_labels = inputs.pop('token_type', None)
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Compute standard Loss
        if type_labels is None:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Then, we compute loss for prior tokens loss
        # log token level loss
        else:
            def _log_per_token_loss (obj: dict, filename: str):
                import json
                with open(os.path.join(self.args.output_dir, filename), "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj) + "\n")
            logits = outputs.logits
            labels = inputs.get('labels', labels)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_type_labels = type_labels[..., 1:].contiguous().to(shift_logits.device)  # Device alignment

            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_type_labels = shift_type_labels.view(-1)

            # Mask for each token type
            prior_mask = (shift_type_labels == PRIOR_TOKEN_TYPE)
            knowledge_mask = (shift_type_labels == KNOWLEDGE_ARTICLE_TOKEN_TYPE)
            nontarget_mask = (shift_type_labels == NONTARGET_ARTICLE_TOKEN_TYPE)
            '''
            loss_fct(shift_logits, shift_labels)
            '''
            # Compute losses
            loss_fct = CrossEntropyLoss(reduction='sum')  # Use 'sum' to handle averages manually
            prior_loss = loss_fct(shift_logits[prior_mask], shift_labels[prior_mask])
            knowledge_loss = loss_fct(shift_logits[knowledge_mask], shift_labels[knowledge_mask])
            nontarget_loss = loss_fct(shift_logits[nontarget_mask], shift_labels[nontarget_mask])

            _log_per_token_loss(
                obj = {
                    "step": self.state.global_step,
                    "epoch": self.state.epoch,
                    "prior_loss_per_token": prior_loss.item() /  max(prior_mask.sum().item(), 1),
                    "knowledge_loss_per_token": knowledge_loss.item() / max(knowledge_mask.sum().item(), 1),
                    "nontarget_loss_per_token": nontarget_loss.item() / max(nontarget_mask.sum().item(), 1)
                    },
                filename = "step_loss.jsonl"
            )
            
            '''This is important'''
            #TODO: If You want to ignore Prior token loss, uncomment the part below
            total_loss = knowledge_loss + nontarget_loss
            loss = total_loss / max(knowledge_mask.sum().item() + nontarget_mask.sum().item(), 1)
        
        return (loss, outputs) if return_outputs else loss