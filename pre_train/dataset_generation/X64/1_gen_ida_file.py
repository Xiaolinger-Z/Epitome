
import os
import subprocess
import glob
from data_gen_config import Config


for bin_path in Config.PORGRAM_ARR:

    for mode in Config.MODE:
        for data_name in Config.NAME:
            paths = glob.glob(Config.O_DIR+ str(os.sep) + bin_path + str(os.sep) + mode + str(os.sep) + data_name + "/*/*")
            
            for file_path in paths:
                if file_path.endswith(".idb") or file_path.endswith(".asm") or file_path.endswith(".i64"):
                    # os.remove(file_path)
                    continue
                if file_path.endswith(".id0") or file_path.endswith(".id1") or file_path.endswith(".id2") or file_path.endswith(".til") or file_path.endswith(".nam"):
                    try:
                        os.remove(file_path)
                    except:
                        print file_path
                
                else:

                    print Config.IDA64_DIR+ " -B \""+ file_path+"\""
                    subprocess.call(Config.IDA64_DIR+ " -B \""+ file_path+"\"", shell=True)
