#!/usr/bin/python3
import json
import os
import sys
import re


class Config:

    # config for function name split and normalize NLP
    MAX_STR_LEN_BEFORE_SEQ_SPLIT = 10
    MIN_MAX_WORD_LEN = 3
    MIN_MAX_ABBR_LEN = 2
    EDIT_DISTANCE_THRESHOLD = 0.5
    WORD_MATCH_THRESHOLD = 0.36787968862663154641
    MAX_WORD_LEN = 13

    # config  for raw data process
    ROOT_DIR = '/root'  # the root path of code
    DATA_ROOT_DIR = ROOT_DIR + '/data'  # the path of dataset
    CODE_DIR = ROOT_DIR + "/Epitome/dataset_generation"

    STEP1_GEN_IDB_FILE = True
    O_DIR = DATA_ROOT_DIR + "raw_data"
    PORGRAM_ARR = ["cross-optimization "]

    MODE = ['X64']
    NAME = ['mix']

    STEP2_GEN_FEA = True
    FEA_DIR = DATA_ROOT_DIR + os.sep + "feature"  # the output path
    IDB_DIR = DATA_ROOT_DIR + os.sep + "raw_data"  # the input path
    SELECT_FEA_DIR = '' # the path of selected feature

    # config for function name tokenization
    unigram_train_path = ""
    Freedom_model  = ""
    Freedom_lexicon_model = ""

    IDA32_DIR = "/opt/idapro-7.3/idat"  # ida pro path
    IDA64_DIR = "/opt/idapro-7.3/idat64"  #

