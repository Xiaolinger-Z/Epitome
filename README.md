# Epitome
Implementation of paper "Epitome"


## Prerequisites
To use Epitome, we need the following tools installed
- IDA Pro 7.3 - for generating the control dependency instructions and data dependency instructions of functions
- python2.7 - the script used by IDA Pro is written in python2.7
- [miasm](https://github.com/cea-sec/miasm) - for converting assembly programs to LLVM IR. We extend it to support more assembly instructions. Please directly copy the `miasm` provided by us to the python directory of `IDA Pro`.
- python3.8 - the scripts used for the training model are written in python3.8
- [NLTK](https://www.nltk.org/), and [SentencePiece](https://github.com/google/sentencepiece)  - for function name preprocessing
- [Gensim](https://radimrehurek.com/gensim/models/word2vec.html) - for label relationship identification.
- We built the other components of Epitome with [Pytorch](https://pytorch.org/) and [Transformer](https://huggingface.co/docs/transformers/v4.18.0/en/index).

## Directory structure
- `pre_train/dataset_generation`: it contains the script to generate a dataset for the assembly language model.
- `pre_train/pre_train_model`: it contains the scripts for training the assembly language model.
- `dataset_generation`:  it contains the scripts to generate a dataset for multi-task learning.
- `training_evalation`: it contains the scripts for training Epitome.

## Table of contents

- [Epitome](#epitome)
  - [Dataset](#dataset)
  - [Pre-train Assembly Language Model](#pre-train-assembly-language-model)
    - [Setup](#setup)
    - [Training](#training)
  - [Model Training](#model-training)
    - [Setup](#setup)
    - [Training](#training)
  - [Prediction and Evaluation](#prediction-and-evaluation)
    - [Function Name Prediction](#function-name-prediction)
    - [Evaluation](#evaluation)
  - [CodeWordNet](#codewordnet)

## Dataset

We provide `x64` dataset under the [`dataset_generation/dataset_sample`](dataset_generation/dataset_sample).



## Pre-train Assembly Language Model
### Setup
1. We need to modify the `pre_train/dataset_generation/data_gen_config.py` file. Simple modification is listed as following, but it need to follow the directory structure we defined:
```
IDA32_DIR = "installation directory of 32-bit IDA Pro program"
IDA64_DIR = "installation directory of 64-bit IDA Pro program"
```
2. We should set ROOT_DIR for the root path of code, and DATA_ROOT_DIR for the path of binaries.
3. We run the `pre_train/dataset_generation/data_gen_command.py` file to generate the control dependency instructions and data dependency instructions for functions.
   
` Note: All steps can be executed in the Linux system.`
### Training
The script for training the model is [`pre_train/pre_train_model/run_pretrain.sh`][pre_train/pre_train_model/run_pretrain.sh], in which you have to set the following parameters:
''' bash
num_train_epochs = 20  # Num of epoch for training 
train_cfg_dataset  = "ROOT_DIR/feature/single_cfg_train_X64.txt"  # The train path of pairs of control dependency instructions
train_dfg_dataset  = "ROOT_DIR/feature/single_dfg_train_X64.txt"  # The train path of pairs of data dependency instructions
test_cfg_dataset  = "ROOT_DIR/feature/single_cfg_test_X64.txt" " # The valid path of pairs of control dependency instructions
test_dfg_dataset = "ROOT_DIR/feature/single_dfg_test_X64.txt"     # The valid path of pairs of data dependency instructions
vocab_path = "./modelout/vocab"  # the path for save vocab
per_device_train_batch_size = 256  # train batch size
per_device_eval_batch_size = 16  # # valid batch size
learning_rate = 5e-5  #learning rate
max_seq_length  = 32  # the length of pairs of instructions    
output_dir  = "./modelout/"  # the model save path
warmup_steps = 10000 # Warmup the learning rate over this many updates


## Model Training

### Setup

#### Start From Scratch
1. We need to modify the `dataset_generation/config.py` file. Simple modification is listed as following, but it need to follow the directory structure we defined:
```
IDA32_DIR = "installation directory of 32-bit IDA Pro program"
IDA64_DIR = "installation directory of 64-bit IDA Pro program"
```
2. We should set ROOT_DIR for the root path of code, and DATA_ROOT_DIR for the path of binaries.
3. We run the `dataset_generation/command.py` file to generate the CFG for functions.
4. We run the `dataset_generation/2.3_process_function_name.py` file for function name preprocessing.
   
` Note: All steps can be executed in the Linux system.`

#### Loade Trained Model
The pretrained model was obtained from [Trex](https://arxiv.org/abs/2012.08680) in Deceber 2021.

### Training
The script for training the model is [`training_evalution/train.py`][training_evalution/train.py], in which you have to set the following parameters in [`training_evalution/model_config.py`][training_evalution/model_config.py]:
''' bash
num_train_epochs = 20  # Num of epoch for training 
train_cfg_dataset  = "ROOT_DIR/feature/single_cfg_train_X64.txt"  # The train path of pairs of control dependency instructions
train_dfg_dataset  = "ROOT_DIR/feature/single_dfg_train_X64.txt"  # The train path of pairs of data dependency instructions
test_cfg_dataset  = "ROOT_DIR/feature/single_cfg_test_X64.txt" " # The valid path of pairs of control dependency instructions
test_dfg_dataset = "ROOT_DIR/feature/single_dfg_test_X64.txt"     # The valid path of pairs of data dependency instructions
vocab_path = "./modelout/vocab"  # the path for save vocab
per_device_train_batch_size = 256  # train batch size
per_device_eval_batch_size = 16  # # valid batch size
learning_rate = 5e-5  #learning rate
max_seq_length  = 32  # the length of pairs of instructions    
output_dir  = "./modelout/"  # the model save path
warmup_steps = 10000 # Warmup the learning rate over this many updates


