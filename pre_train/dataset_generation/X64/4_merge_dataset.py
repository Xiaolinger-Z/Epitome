import os
from data_gen_config import Config
from sklearn.model_selection import train_test_split

def findfiles(path):
    file_list = os.listdir(path)
    cfg_file_list = []

    for file_name in file_list:

        cur_file = os.path.join(path, file_name)
        if os.path.isdir(cur_file):
            cfg_files = findfiles(cur_file)
            cfg_file_list.extend(cfg_files)
        else:

            if '_cfg_' in file_name:
                # print file_name
                cfg_file_list.append(cur_file)
    return cfg_file_list


def read_files(path):
    data_dict = dict()
    with open(path, 'r') as f_cfg:
        cfg_out = f_cfg.readlines()
        for i in cfg_out:
            tmp = i.strip().split(';')
            name = tmp[0]
            data = tmp[1:]
            if len(data) > 1:
                if name not in data_dict.keys():
                    data_dict[name] = data
                else:
                    print(name)

    return data_dict


def write_files(path, data):
    with open(path, 'a') as f:
        for values in data.values():
            for v in values:
                if len(v) > 1:
                    f.write(v + '\n')


def aligen_datasets(cfg_file_list, dfg_file_list, cfg_outfn, dfg_outfn):

    for cur_cfg_file, cur_dfg_file in zip(cfg_file_list, dfg_file_list):

        cfg_data = dict()
        dfg_data = dict()

        cfg_tmp_data = read_files(cur_cfg_file)
        dfg_tmp_data = read_files(cur_dfg_file)
        # print('!!!!!!!!!!!!!')
        # print(len(og_tmp_data))
        for key in dfg_tmp_data.keys():
            if key in cfg_tmp_data.keys():
                min_len = min(len(cfg_tmp_data[key]), len(dfg_tmp_data[key]))
                cfg_data[key] = cfg_tmp_data[key][:min_len]
                dfg_data[key] = dfg_tmp_data[key][:min_len]

        # print(len(og_data))

        write_files(cfg_outfn, cfg_data)
        write_files(dfg_outfn, dfg_data)

if __name__ == '__main__':
    final_cfg_list, final_dfg_list = [], []

    trainset_path = ""
    training_binary_name =[]
    with open(trainset_path, 'r') as ft:
        for name in ft.readlines():
            training_binary_name.append(name.strip())

    for program in Config.PORGRAM_ARR:
        for mode in Config.MODE:
            for name in Config.NAME:
                cfg_list = findfiles(Config.FEA_DIR + os.sep + program + os.sep + mode + os.sep + name)
                final_cfg_list.append(cfg_list)

    for cur_file_list, name in zip(final_cfg_list, Config.NAME):

        cfg_files, dfg_files = [], []

        for cur_file in cur_file_list:
            base_path = os.path.dirname(cur_file)
            dfg_name = os.path.basename(cur_file).replace('_cfg_', '_dfg_')
            tmp_name = os.path.basename(cur_file).split('_')[:-3]
            if '_'.join(tmp_name) in training_binary_name:
                cfg_files.append(cur_file)
                dfg_files.append(base_path + '/' + dfg_name)

        cfg_train, cfg_dev, dfg_train, dfg_dev = train_test_split(cfg_files, dfg_files, test_size=0.05,
                                                                  random_state=0)

        cfg_train_outfn = Config.FEA_DIR + os.sep + Config.PORGRAM_ARR[0] + os.sep + Config.MODE[0] + os.sep + \
                          "single_cfg_train_{}.txt".format(name)
        dfg_train_outfn = Config.FEA_DIR + os.sep + Config.PORGRAM_ARR[0] + os.sep + Config.MODE[0] + os.sep + \
                          "single_dfg_train_{}.txt".format(name)

        aligen_datasets(cfg_train, dfg_train, cfg_train_outfn, dfg_train_outfn)

        cfg_val_outfn = Config.FEA_DIR + os.sep + Config.PORGRAM_ARR[0] + os.sep + Config.MODE[0] + os.sep + \
                        "single_cfg_val_{}.txt".format(name)
        dfg_val_outfn = Config.FEA_DIR + os.sep + Config.PORGRAM_ARR[0] + os.sep + Config.MODE[0] + os.sep + \
                        "single_dfg_val_{}.txt".format(name)

        aligen_datasets(cfg_dev, dfg_dev, cfg_val_outfn, dfg_val_outfn)



