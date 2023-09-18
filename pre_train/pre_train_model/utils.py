from torch import nn
import torch
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,  f1_score, fbeta_score
from sklearn.model_selection import train_test_split

class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.
    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100
    def __init__ (self, epsilon = 0.1, ignore_index=-100):
        # super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss



class NextPrediction:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.
    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    ignore_index: int = -100
    def __init__ (self, ignore_index=-100):
        # super().__init__()
        self.loss_func = nn.NLLLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        loss = self.loss_func(logits, labels)
        return loss


class EvalLoopOutput(NamedTuple):
    predictions: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


def compute_metrics(p):
    preds, labels = p
    # _, predictions = torch.max(predictions, -1)
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    accuracy = accuracy_score(labels, preds, normalize=True)

    return {
        "accuracy": accuracy,
    }


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]

    return logits.argmax(dim=-1)