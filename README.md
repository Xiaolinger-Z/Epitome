# Epitome
Implementation of paper "Epitome"


## Prerequisites
To use Epitome, we need the following tools installed
- IDA Pro - for generating the LSFG （data flow graph and control flow graph）and extracting features of basic blocks
- python2.7 - all the source code is written in python2.7
- [miasm](https://github.com/cea-sec/miasm) - for converting assembly programs to LLVM IR. We extend it to support more assembly instructions. Please directly copy the `miasm` provided by us to the python directory of `IDA Pro`.
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
  -  - [Setup](#setup)
    - [Training](#training)
  - [Model Training](#model-training)
    - [Setup](#setup)
    - [Training](#training)
  - [Prediction and Evaluation](#prediction-and-evaluation)
    - [Function Name Prediction](#function-name-prediction)
    - [Evaluation](#evaluation)
  - [CodeWordNet](#codewordnet)

## Dataset

We provide a sample `x64` dataset under the [`dataset_generation/dataset_sample`](dataset_generation/dataset_sample) directory and its binarization result under the[`data_bin`](data_bin) directory.

For more details on how these datasets are generated from binaries, please refer to the README under [`dataset_generation/`](dataset_generation/).


## Pre-train Assembly Language Model
### Setup
1. We need modify the `pre_train/dataset_generation/data_gen_config.py` file. Simple modification is listed as following, but it need to follow the directory structure we defined:
```
IDA32_DIR = "installation directory of 32-bit IDA Pro program"
IDA64_DIR = "installation directory of 64-bit IDA Pro program"
```
2. We should set ROOT_DIR for the root path of code, and DATA_ROOT_DIR for the path of binaries.
3. We run the `pre_train/dataset_generation/data_gen_command.py` file to generate the control dependency instructions and data dependency instructions for functions.
   
` Note: All steps can be executed in the Linux system.`
### Training



