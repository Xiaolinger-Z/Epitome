#!/usr/bin/python
# -*- coding: UTF-8 -*-
from data_gen_config import Config
import os
import subprocess
import glob

if Config.STEP1_GEN_IDB_FILE:
    print "step1. convert to idb file"
    subprocess.call(["python", Config.CODE_DIR+ os.sep + "1_gen_ida_file.py"])

if Config.STEP2_GEN_FEA:
    print "step2-1. generate feature file"

    # if os.path.exists(config.FEA_DIR):
    #     shutil.rmtree(config.FEA_DIR)
    if not os.path.exists(Config.FEA_DIR):
        os.mkdir(Config.FEA_DIR)

    for program in Config.PORGRAM_ARR:
       for mode in Config.MODE:
            for data_name in Config.NAME:

                temp_idb = Config.IDB_DIR + str(os.sep) + program+ str(os.sep) + mode + str(os.sep) + data_name 

                for version in os.listdir(temp_idb):
                    curFeaDir = Config.FEA_DIR + str(os.sep) + str(program) + str(os.sep) + mode + str(os.sep) + data_name + str(os.sep) + str(version)

                    curBinDir = temp_idb + str(os.sep) + str(version)

                    if not os.path.exists(curFeaDir):
                        os.makedirs(curFeaDir)

                    filters = glob.glob(curBinDir + os.sep + "*.idb")
                    filters = filters + (glob.glob(curBinDir + os.sep + "*.i64"))
                        
                    for i in filters:

                        if i.endswith("idb"):
                            
                            print Config.IDA32_DIR+" -A -S\""+Config.CODE_DIR+ os.sep + "2_gen_bert_cfg_dataset.py "+curFeaDir+"  "+i +"  "+ "\"  "+i+"\n\n"
                            os.popen("TVHEADLESS=1 " + Config.IDA32_DIR+" -S\""+Config.CODE_DIR+ os.sep + "2_gen_bert_cfg_dataset.py "+curFeaDir+" "+ i + " " + "\"  "+i)

                            print Config.IDA32_DIR + " -A -S\"" + Config.CODE_DIR + os.sep + "3_gen_bert_dfg_dataset.py " + curFeaDir + "  " + i + "  " + "\"  " + i + "\n\n"
                            os.popen("TVHEADLESS=1 " + Config.IDA32_DIR + " -S\"" + Config.CODE_DIR + os.sep + "3_gen_bert_dfg_dataset.py " + curFeaDir + " " + i + " " + "\"  " + i)

                        else:
                            print Config.IDA64_DIR+" -A -S\""+Config.CODE_DIR+ os.sep + "2_gen_bert_cfg_dataset.py "+curFeaDir+"  "+i +"  "+ "\"  "+i+"\n\n"
                            os.popen("TVHEADLESS=1 " + Config.IDA64_DIR+" -A -S\""+Config.CODE_DIR+ os.sep + "2_gen_bert_cfg_dataset.py "+curFeaDir+" "+ i + " " + "\"  "+i)

                            print Config.IDA64_DIR + " -A -S\"" + Config.CODE_DIR + os.sep + "3_gen_bert_dfg_dataset.py " + curFeaDir + "  " + i + "  " + "\"  " + i + "\n\n"
                            os.popen("TVHEADLESS=1 " + Config.IDA64_DIR + " -A -S\"" + Config.CODE_DIR + os.sep + "3_gen_bert_dfg_dataset.py " + curFeaDir + " " + i + " " + "\"  " + i)
