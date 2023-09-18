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
#### Start From Scratch
1. We need to modify the `pre_train/dataset_generation/data_gen_config.py` file. Simple modification is listed as following, but it need to follow the directory structure we defined:
```
IDA32_DIR = "installation directory of 32-bit IDA Pro program"
IDA64_DIR = "installation directory of 64-bit IDA Pro program"
```
2. We should set ROOT_DIR for the root path of code, and DATA_ROOT_DIR for the path of binaries.
3. We run the `pre_train/dataset_generation/data_gen_command.py` file to generate the control dependency instructions and data dependency instructions for functions.

```bash
cd pre_train/dataset_generation/
python2 data_gen_command.py
python3 4_merge_dataset.py
```

   
` Note: All steps can be executed in the Linux system.`

#### Loade Trained Model
The pretrained model was obtained from [Trex](https://arxiv.org/abs/2012.08680) in Deceber 2021.

### Training
The script for training the model is [`pre_train/pre_train_model/run_pretrain.sh`][pre_train/pre_train_model/run_pretrain.sh], in which you have to set the following parameters:

```bash
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
'''

```bash
cd pre_train/pre_train_model/
bash run_pretrain.sh
```

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

```bash
cd dataset_generation/
python2 command.py
python3 2.3_process_function_name.py
```

In order to speed up the training process, we  binarize the dataset. You have to set the following parameters:

''' bash
  datapre = '' # location of the data corpus and test dataset
  inst_vocab_path  = '../pre_train/pre_train_model/modelout/vocab' #assembly instruction vocab path which is pretrain model generate
  label_vocab_path = './modelout/label_vocab'  # function name vocab path
  max_label_num = 10 # function name length
  node_len = 16  # instruction length
  node_num = 256  # the number of node in fined-grained CFG
'''


```bash
cd training_evalution
python3 generate_dataset.py
```


#### Loade Trained Model
The pretrained model was obtained from [Trex](https://arxiv.org/abs/2012.08680) in Deceber 2021.

### Training


The script for training the model is [`training_evalution/train.py`](training_evalution/train.py), in which you have to set the following parameters in [`training_evalution/model_config.py`](training_evalution/model_config.py):

```bash
  datapre = '' # location of the data corpus and test dataset
  test_datapre =''  # location of the test dataset
  node_vocab_path  = '../pre_train/pre_train_model/modelout/vocab' #assembly instruction vocab path which is pretrain model generate
  graph_label_vocab_path = ''  # function name vocab path
  pre_train_model ='../pre_train/pre_train_model/modelout/best_ceckpoint' # pre-train model path
  word_net_path = 'X64_wordnet.json' 
  save_path = './modelout/'
  test_opt = ['O0', 'O1', 'O2', 'O3', 'Os'] # compilation optimizations for tesing 
  lr = 1e-4  # Initial learning rate
  min_lr = 1e-6 # Minimum learning rate.
  dropout = 0.1  # Dropout rate (1 - keep probability)
  target_len = 10 # function name length
  node_len = 16  # instruction length
  node_num = 256  # the number of node in fined-grained CFG
  num_blocks = 6 # the num block of transformer
  num_heads = 8 # the num head of transformer
  batch_size = 64  # Input batch size for training
  epochs = 30 #  Number of epochs to train
  emb_dim = 128  # Size of embedding
  conv_feature_dim = 128 # Size of conv layer in node embedding
  hidden_dim = 256 # Size of hidden size
  radius = 2  # Diameter of ego networks
  beam = 3 # 'beam size'
  factor = 0.5 # Factor in the ReduceLROnPlateau learning rate scheduler
  patience = 3 #Patience in the ReduceLROnPlateau learning rate scheduler
```

To train the model, run the `train.py`
```bash
cd training_evalution
python3 train.py
```


  


