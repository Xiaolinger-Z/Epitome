#!/usr/bin/python3
import os
import sys
import re


class Config:

    # config  for raw data process
    ROOT_DIR = ''# the root path of code
    DATA_ROOT_DIR = '' #  the path of dataset
    CODE_DIR = ROOT_DIR

    STEP1_GEN_IDB_FILE = True
    O_DIR = DATA_ROOT_DIR + "raw_data"
    PORGRAM_ARR=["cross-optimization "]

    MODE = ['X64']
    NAME = ['O0','O1','O2', 'O3', 'Os']

    STEP2_GEN_FEA = True
    FEA_DIR = DATA_ROOT_DIR+ os.sep + "feature" # the output path
    IDB_DIR = DATA_ROOT_DIR+ os.sep + "raw_data" # the input path


    IDA32_DIR = "/opt/idapro-7.3/idat"# ida pro path
    IDA64_DIR = "/opt/idapro-7.3/idat64"#"




