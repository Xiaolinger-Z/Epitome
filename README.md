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
- `training_evaluation`: it contains the scripts for training Epitome.
- `label_relationship_identify`: it contains the scripts for function name processing. 

## Table of contents

- [Epitome](#epitome)
  - [Dataset](#dataset)
  - [Pre-train Assembly Language Model](#pre-train-assembly-language-model)
    - [Setup](#setup)
    - [Training](#training)
    - [Trained Model](#trained-model)
  - [Model Training](#model-training)
    - [Setup](#setup)
    - [Training](#training)
    - [Trained Model](#trained-model)
  - [Prediction](#prediction)
  - [LabelRelationshipIdentify](#label-relationship-identify)

## Dataset
To test the model, please first download the [processed dataset](https://drive.google.com/file/d/1bIDtujLbo2v7hoowGpZNiOv5KkhQnY1H/view?usp=sharing) and put it under the [dataset](dataset) directory.
Because the training data is too large, it is difficult to upload now and will be made public later.

## Pre-train Assembly Language Model
### Setup
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

### Training
The script for training the model is [`pre_train/pre_train_model/run_pretrain.sh`](pre_train/pre_train_model/run_pretrain.sh), in which you have to set the following parameters:

```bash
num_train_epochs = 20  # Num of epoch for training 
train_cfg_dataset  = "ROOT_DIR/feature/single_cfg_train_X64.txt"  # The train path of pairs of control dependency instructions
train_dfg_dataset  = "ROOT_DIR/feature/single_dfg_train_X64.txt"  # The train path of pairs of data dependency instructions
test_cfg_dataset  = "ROOT_DIR/feature/single_cfg_test_X64.txt"  # The valid path of pairs of control dependency instructions
test_dfg_dataset = "ROOT_DIR/feature/single_dfg_test_X64.txt"     # The valid path of pairs of data dependency instructions
vocab_path = "./modelout/vocab"  # the path for save vocab
per_device_train_batch_size = 256  # train batch size
per_device_eval_batch_size = 16  # # valid batch size
learning_rate = 5e-5  #learning rate
max_seq_length  = 32  # the length of pairs of instructions    
output_dir  = "./modelout/"  # the model save path
warmup_steps = 10000 # Warmup the learning rate over this many updates
```

To train the model, run the `run_pretrain.sh`

```bash
cd pre_train/pre_train_model/
bash run_pretrain.sh
```

### Trained Model
The pretrained model was obtained from [pretrained model](https://drive.google.com/drive/folders/1R6neL8T2Rknm8T95p3bST9dehreKGc-m?usp=sharing) and the assembly instructions vocab was obtained from [vocab](https://drive.google.com/file/d/1jIETwM2slYPe5Ob3adTTnAVoAAyvk6lK/view?usp=sharing). You can put them under the [pre_train/pre_train_model/modelout](pre_train/pre_train_model/modelout) directory.

## Model Training

### Setup

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

```bash
  datapre = '' # location of the data corpus and test dataset
  inst_vocab_path  = '../pre_train/pre_train_model/modelout/vocab' #assembly instruction vocab path which is pretrain model generate
  label_vocab_path = './modelout/label_vocab'  # function name vocab path
  max_label_num = 10 # function name length
  node_len = 16  # instruction length
  node_num = 256  # the number of node in fined-grained CFG
```

To binarize the dataset, run the `generate_dataset.py`

```bash
cd training_evaluation
python3 generate_dataset.py
```

### Training


The script for training the model is [`training_evaluation/train.py`](training_evaluation/train.py), in which you have to set the following parameters in [`training_evaluation/model_config.py`](training_evaluation/model_config.py):

```bash
  datapre = '../dataset/X64_dataset' # location of the data corpus and test dataset
  node_vocab_path  = '../pre_train/pre_train_model/modelout/vocab' #assembly instruction vocab path which is pretrain model generate
  graph_label_vocab_path = './modelout/label_vocab'  # function name vocab path
  pre_train_model ='../pre_train/pre_train_model/modelout/best_ceckpoint' # pre-train model path
  save_path = './modelout/'
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
  beam = 0 # 'beam size'
  factor = 0.5 # Factor in the ReduceLROnPlateau learning rate scheduler
  patience = 3 #Patience in the ReduceLROnPlateau learning rate scheduler
```

To train the model, run the `train.py`

```bash
cd training_evaluation
python3 train.py
```

### Trained Model
The pretrained model was obtained from [pretrained model](https://drive.google.com/file/d/1QsxoRSSVlDDidasTu4GRfNRQNwjJiqcD/view?usp=sharing) and the label vocab was obtained from [vocab](https://drive.google.com/file/d/1fAZaJvUhmiv46ni21aZylUINNE6UPWNx/view?usp=sharing).
You can put them under the [training_evaluation/modelout](training_evalution/modelout) directory.

  
## Prediction

The script for training the model is [`training_evaluation/test.py`](training_evaluation/test.py), in which you have to set the following parameters in [`training_evaluation/model_config.py`](training_evaluation/model_config.py):

```bash
  test_datapre ='../dataset/test_X64_dataset'  # location of the test dataset
  node_vocab_path  = '../pre_train/pre_train_model/modelout/vocab' #assembly instruction vocab path which is pretrain model generate
  graph_label_vocab_path = './modelout/label_vocab'  # function name vocab path
  pre_train_model ='../pre_train/pre_train_model/modelout/best_ceckpoint' # pre-train model path
  word_net_path = 'X64_wordnet.json' 
  save_path = './modelout/'
  test_opt = ['O0', 'O1', 'O2', 'O3', 'Os'] # compilation optimizations for testing 
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
  beam = 0 # 'beam size'
```

The predicted names are saved in the [`training_evaluation/modelout/prediction`](training_evaluation/modelout/prediction) directory. Note that, the result of the evaluation is printed.


## LabelRelationshipIdentify

We address the challenges of the noisy nature of natural languages, we propose to generate the distributed representation of function name words and calculate the semantic distance to identify the relationship of labels. We provide the script [`label_relationship_identify/train_model.py`](label_relationship_identify/train_model.py) for label embedding model training.


