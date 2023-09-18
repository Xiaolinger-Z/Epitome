import torch
import logging
import os
import sys
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments
from transformers import BertConfig, BertForPreTraining, BertModel
import transformers
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from transformers import CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, HfArgumentParser
from transformers.trainer_utils import is_main_process
from transformers.optimization import AdamW
from language_model import BERTInstsLM
from trainer import BERTInstsTrainer
from dataset import WordVocab, BARTDataset, DataCollatorForDenoisingLM
from utils import compute_metrics, preprocess_logits_for_metrics
from optim_schedule import ScheduledOptim
from torch.optim import Adam, AdamW
import random
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default= None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )

    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    train_cfg_dataset: Optional[str] = field(
        default="",
    )

    train_dfg_dataset: Optional[str] = field(
        default=""
    )

    test_cfg_dataset: Optional[str] = field(
        default="",
    )

    test_dfg_dataset: Optional[str] = field(
        default=""
    )

    vocab_path: Optional[str] = field(
        default="",
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.

    num_train_epochs: int = field(
        default=20,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    do_train :bool=field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )
    do_eval:bool=field(
        default= False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )


    ##########################################################################
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    output_dir: str = field(
        default="./modelout/",
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    ##########################################################################

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device

class Config:

    max_length = 512
    emb_size = 128
    hidden_size =128
    n_layers =12
    n_head =8
    dropout=0.0

    eval_batch_size = 8

    warmup_steps = 0

    seed = 42

def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.do_eh_loss:
        if model_args.eh_loss_margin is None or model_args.eh_loss_weight is None:
            parser.error('Requiring eh_loss_margin and eh_loss_weight if do_eh_loss is provided')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    if  os.path.exists(data_args.vocab_path):
        print("Loading Vocab", data_args.vocab_path)
        tokenizer = WordVocab.load_vocab(data_args.vocab_path)
        print("Vocab Size: ", len(tokenizer))
    else:
        tokenizer = WordVocab([data_args.train_cfg_dataset, data_args.train_dfg_dataset, data_args.test_dfg_dataset, data_args.test_cfg_dataset],
                                   model_max_length=data_args.max_seq_length, max_size=13000, min_freq=1)
        print("VOCAB SIZE:", len(tokenizer))
        tokenizer.save_vocab(data_args.vocab_path)


    train_datasets = BARTDataset(data_args.train_cfg_dataset, data_args.train_dfg_dataset, tokenizer, seq_len=data_args. max_seq_length,
                                           corpus_lines=None, on_memory=True)

    # tokenized_train_datasets = group_texts(train_datasets, 1024)
    val_datasets = BARTDataset(data_args.test_cfg_dataset, data_args.test_dfg_dataset, tokenizer, seq_len = data_args. max_seq_length, on_memory=True) \
        if data_args.test_cfg_dataset is not None else None

    # Initialize our training

    vocab_size = len(tokenizer)

    # Data collator
    # This one will take care of randomly masking the tokens and permuting the sentences.
    data_collator = DataCollatorForDenoisingLM(
        tokenizer=tokenizer,
        max_length=data_args.max_seq_length,
    )

    config = BertConfig(hidden_size = 128, num_hidden_layers = 12,num_attention_heads = 8, intermediate_size = 512,
                        vocab_size=vocab_size, attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0)
    print(config)

    bertmodel = BERTInstsLM(config).cuda()  # to(self.device)

    optimizer = AdamW(bertmodel.parameters(), lr=training_args.learning_rate,
                                  betas=(0.9, 0.999), weight_decay=training_args.weight_decay)
    optim_schedule = ScheduledOptim(optimizer,  config.hidden_size, n_warmup_steps=training_args.warmup_steps)

    trainer = BERTInstsTrainer(
        model=bertmodel,
        args=training_args,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset = val_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics = compute_metrics,
        optimizers = (optimizer, optim_schedule),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        #gradient_accumulation_steps=8,
    )
    trainer.model_args = model_args

    if training_args.do_train:
        checkpoint = None
        if model_args.model_name_or_path is not None:
            checkpoint = model_args.model_name_or_path
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results

if __name__ == "__main__":
    main()