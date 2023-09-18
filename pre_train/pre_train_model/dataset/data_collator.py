import torch
import math
import numpy as np
from typing import List, Dict
from itertools import chain

class DataCollatorForDenoisingLM:
    

    def __init__(self, tokenizer, max_length, mask_ratio = 0.3, poisson_lambda = 3.0, permute_sentence_ratio = 0.0):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.poisson_lambda = poisson_lambda
        self.permute_sentence_ratio = permute_sentence_ratio


    def __call__(self, examples: List):
        # convert list to dict and tensorize input

        examples =  {k: [examples[i][k] for i in range(len(examples))] for k, v in examples[0].items()}

        cfg_batch= self.tokenizer.pad(examples['cfg_bert_input'], return_tensors='np', max_length=self.max_length, padding='max_length')
        dfg_batch = self.tokenizer.pad(examples['dfg_bert_input'], return_tensors='np', max_length=self.max_length, padding='max_length')


        cfg_batch["labels"] = cfg_batch["input_ids"].copy()

        # permuting sentences
        do_permute = False
        if self.permute_sentence_ratio > 0.0:
            cfg_batch["input_ids"] = self.permute_sentences(cfg_batch["input_ids"])
            #batch["token_type_ids"] = self.permute_sentences(cfg_batch["token_type_ids"])
            do_permute = True

        # masking span of tokens (text infilling in the paper)
        if self.mask_ratio:
            cfg_batch["input_ids"], cfg_batch["labels"] = self.span_mask_tokens(
                cfg_batch["input_ids"], cfg_batch["labels"], do_permute
            )

        # ignore pad tokens
        # batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_index).astype(int)
        output = {"dfg_input_ids": dfg_batch['input_ids'],
                  "dfg_token_type_ids": dfg_batch['token_type_ids'],
                  "dfg_attention_mask": dfg_batch["attention_mask"],
                  "dfg_is_next": examples['dfg_is_next'],
                  "input_ids": cfg_batch['input_ids'],
                  "token_type_ids": cfg_batch['token_type_ids'],
                  "attention_mask": cfg_batch["attention_mask"],
                  "labels": cfg_batch["labels"],
                  "cfg_is_next": examples['cfg_is_next']
                  }
        return {key: torch.tensor(value) for key, value in output.items()}
        #return batch

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.max_length:
            total_length = (total_length // self.max_length) * self.max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + self.max_length] for i in range(0, total_length, self.max_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def permute_sentences(self, input_ids):
        """
        Shuffle sentences in each document.
        """
        results = input_ids.copy()

        # find end locations of sentences
        end_sentence_mask = input_ids == self.tokenizer.eos_index
        sentence_ends = np.argwhere(end_sentence_mask)
        sentence_ends[:, 1] += 1
        example_has_multiple_sentences, num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)
        num_sentences_map = {sent_idx: count for sent_idx, count in zip(example_has_multiple_sentences, num_sentences)}

        num_to_permute = np.ceil(num_sentences * self.permute_sentence_ratio).astype(int)
        num_to_permute_map = {
            sent_idx: count for sent_idx, count in zip(example_has_multiple_sentences, num_to_permute)
        }

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(sentence_ends[:, 0], return_index=True)[1][1:])
        sentence_ends_map = {sent_idx: count for sent_idx, count in zip(example_has_multiple_sentences, sentence_ends)}

        for i in range(input_ids.shape[0]):
            if i not in example_has_multiple_sentences:
                continue
            substitutions = np.random.permutation(num_sentences_map[i])[: num_to_permute_map[i]]
            ordering = np.arange(0, num_sentences_map[i])
            ordering[substitutions] = substitutions[np.random.permutation(num_to_permute_map[i])]

            # write shuffled sentences into results
            index = 0
            for j in ordering:
                sentence = input_ids[i, (sentence_ends_map[i][j - 1] if j > 0 else 0): sentence_ends_map[i][j]]

                results[i, index: index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def span_mask_tokens(self, input_ids, labels, do_permute):
        """
        Sampling text spans with span lengths drawn from a Poisson distribution and masking them.
        """
        special_tokens_mask_labels = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask_inputs = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_ids.tolist()
        ]
        special_tokens_mask_labels = np.array(special_tokens_mask_labels, dtype=bool)
        special_tokens_mask_inputs = np.array(special_tokens_mask_inputs, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token_mask = ~(input_ids == self.tokenizer.pad_index) & ~special_tokens_mask_inputs
        num_tokens_to_mask = int(math.ceil(is_token_mask.astype(float).sum() * self.mask_ratio))
        if num_tokens_to_mask == 0:
            return input_ids, labels

        # generate a sufficient number of span lengths
        span_lengths = np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))
        while np.cumsum(span_lengths, 0)[-1] < num_tokens_to_mask:
            span_lengths = np.concatenate(
                [span_lengths, np.random.poisson(lam=self.poisson_lambda, size=(num_tokens_to_mask,))]
            )

        # remove all spans of length 0
        # note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        span_lengths = span_lengths[span_lengths > 0]

        # trim to about num_tokens_to_mask tokens
        cutoff_idx = np.argmin(np.abs(np.cumsum(span_lengths, 0) - num_tokens_to_mask)) + 1
        span_lengths = span_lengths[:cutoff_idx]

        # randomly choose starting positions for masking
        token_indices = np.argwhere(is_token_mask == 1)
        span_starts = np.random.permutation(token_indices.shape[0])[: span_lengths.shape[0]]
        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(input_ids, fill_value=False)

        # mask starting positions
        for mi in masked_indices:
            mask[tuple(mi)] = True
        span_lengths -= 1

        # fill up spans
        max_index = input_ids.shape[1] - 1
        remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            span_lengths -= 1
            remaining = (span_lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask_inputs)] = False
        input_ids[np.where(mask)] = self.tokenizer.mask_index
        if not do_permute:
            labels[np.where(mask == 0)] = -100
        else:
            labels[np.where(special_tokens_mask_labels)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_input_ids = np.full_like(input_ids, fill_value=self.tokenizer.pad_index)
        for i, example in enumerate(input_ids):
            new_example = example[~to_remove[i]]
            new_input_ids[i, : new_example.shape[0]] = new_example

        return new_input_ids, labels


def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    """Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by
    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned."""
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx

def group_texts(examples, max_seq_length):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
