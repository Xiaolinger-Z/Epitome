import os
from packaging import version
from transformers import Trainer, PreTrainedModel
from transformers.file_utils import is_apex_available
from transformers.trainer import WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model
from transformers.utils import logging
import torch
import torch.nn as nn
from utils import LabelSmoother, NextPrediction
from typing import Optional

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"

class BERTInstsTrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dfg_next_criterion = NextPrediction()
        self.cfg_next_criterion = NextPrediction()
        self.mf_criterion = LabelSmoother()

    ##########################################################################
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        if "labels" in inputs:
            labels = inputs.pop("labels")
            dfg_is_next = inputs.pop("dfg_is_next")
            cfg_is_next = inputs.pop("cfg_is_next")

        else:
            labels = None
            dfg_is_next = None
            cfg_is_next = None

        dfg_next_sent_output, cfg_next_sent_output, mf_lm_output = model(**inputs)
        outputs = mf_lm_output
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            dfg_next_loss = self.dfg_next_criterion(dfg_next_sent_output, dfg_is_next)
            cfg_next_loss = self.cfg_next_criterion(cfg_next_sent_output, cfg_is_next)
            mf_loss = self.mf_criterion(mf_lm_output, labels)
            loss = mf_loss + dfg_next_loss + cfg_next_loss

        else:
            loss = None


        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                if state_dict is None:
                    state_dict = self.model.state_dict()
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = self.model.state_dict()
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    ##########################################################################



