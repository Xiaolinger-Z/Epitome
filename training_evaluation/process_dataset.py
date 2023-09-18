import os
import sys
import pickle
import numpy as np
import glob
import networkx as nx
import random
from graph_label_vocab import WordVocab
sys.path.append('../pre_train/pre_train_code')
import dataset
import csv
import codecs
from collections import Counter
from re import compile, VERBOSE

attrListInfo = ['Func_PathList'] # the max length is 13
#22

attrValueInfo =  ['Func_AvePath', 'Func_AveInDegree', 'Func_AveOutDegree', 'Func_AveAllDegree', 'Func_MaxInDegree',
                     'Func_MaxOutDegree', 'Func_MaxAllDegree', 'Func_Node', 'Func_Edge',
                     'Func_AllEtp', 'Func_Jmp', 'Func_Jmpp',
                     'Func_CodeSize', 'Func_InstSize', 'Func_Density', 'Func_Diameter', 'Func_CallFromNums',
                     'Func_CallFromTimes', 'Func_CallToNums', 'Func_CallToTimes']


#20
'''
attrValueInfo =  ['Func_AvePath', 'Func_AveAllDegree', 'Func_MaxInDegree',
                     'Func_MaxOutDegree', 'Func_MaxAllDegree', 'Func_Node', 'Func_Edge',
                     'Func_AllEtp', 'Func_Jmp', 'Func_Jmpp',
                     'Func_CodeSize', 'Func_InstSize', 'Func_Density', 'Func_Diameter', 'Func_CallFromNums',
                     'Func_CallFromTimes', 'Func_CallToNums', 'Func_CallToTimes']
'''
# 18 0.3
'''
attrValueInfo =  ['Func_AvePath', 'Func_AveAllDegree', 'Func_MaxInDegree',
                     'Func_MaxOutDegree', 'Func_MaxAllDegree', 'Func_Node', 'Func_Edge',
                     'Func_Jmp', 
                     'Func_CodeSize', 'Func_InstSize', 'Func_Density', 'Func_Diameter', 'Func_CallFromNums',
                     'Func_CallFromTimes', 'Func_CallToNums', 'Func_CallToTimes']
'''
# 16 0.5
'''
attrValueInfo =  ['Func_AvePath', 'Func_MaxInDegree',
                     'Func_MaxOutDegree', 'Func_MaxAllDegree', 'Func_Node', 'Func_Edge',
                     'Func_Jmp', 
                     'Func_CodeSize', 'Func_InstSize', 'Func_Diameter', 'Func_CallFromNums',
                     'Func_CallFromTimes', 'Func_CallToNums', 'Func_CallToTimes']
'''

attrSetInfo = ['Func_StrList'] # the max length is 12


RE_WORDS = compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+  # Numbers
    ''', VERBOSE)

def get_proc_str_list(str_list, node_vocab, feature_num):
    # get sub-tokens. Remove them if len()<1 (single letter or digit)
    str_list = str_list.replace('[', '').replace(']', '').strip()
    str_list = str_list.split(',')
    outs =[]
    for s in str_list:
        s = s.strip()
        for x in RE_WORDS.findall(s):
            if len(x)>1 and len(x)<14:
                outs.append(x.lower())

    output = node_vocab.to_seq(' '.join(outs), feature_num)
    return output

def save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))

def get_split_precent(file_list):

    n_total = int(len(file_list))
    offset0 = int(n_total * 0.8)
    offset1 = int(n_total * 0.1)
    offset2 = int(n_total * 0.1)

    if n_total == 0:
        return []

    if offset0 + offset1 + offset2 > n_total:
        print("error len overflow")
        return 0

    if offset0 + offset1 + offset2 <= n_total:
        random.shuffle(file_list)

    train_data = file_list[:offset0]
    valid_data = file_list[offset0:offset0 + offset1]
    test_data = file_list[offset0 + offset1:]
    return train_data, valid_data, test_data

def get_dataset_split(datapre, outpath, arch='X86'):
    file_list_O0 = []
    file_list_O1 = []
    file_list_O2 = []
    file_list_O3 = []
    file_list_Os = []

    for idx, datadir_name in enumerate(os.listdir(datapre)):
        tmp_path = os.path.join(datapre, datadir_name)
        temp_filters = glob.glob(tmp_path + os.sep + "*_cfg_A.txt")
        if 'O0' in datadir_name:
            file_list_O0.extend(temp_filters)
        elif 'O1' in datadir_name:
            file_list_O1.extend(temp_filters)
        elif 'O2' in datadir_name:
            file_list_O2.extend(temp_filters)
        elif 'O3' in datadir_name:
            file_list_O3.extend(temp_filters)
        else:
            file_list_Os.extend(temp_filters)

    train_datas =[]
    valid_datas =[]
    test_datas =[]

    for dataset in [file_list_O0, file_list_O1, file_list_O2, file_list_O3, file_list_Os]:
        tmp_train, tmp_val, tmp_test = get_split_precent(dataset)
        train_datas.extend(tmp_train)
        valid_datas.extend(tmp_val)
        test_datas.extend(tmp_test)

    split_dict = {}
    split_dict["train"] = train_datas
    split_dict["valid"] = valid_datas
    split_dict["test"] = test_datas


    for split in ['train', 'valid', 'test']:
        list_path = outpath + os.sep + f"{split}_{arch}_list.txt"
        with open(list_path, 'w') as f:
            for line in split_dict[split]:
                f.write(line+'\n')


def get_dataset_binary_split(datapre, outpath, arch='X86'):
    file_list_O0 = []
    file_list_O1 = []
    file_list_O2 = []
    file_list_O3 = []
    file_list_Os = []


    for idx, datadir_name in enumerate(os.listdir(datapre)):
        tmp_path = os.path.join(datapre, datadir_name)
        temp_filters = glob.glob(tmp_path + os.sep + "*_cfg_A.txt")
        for tmp in temp_filters:
            if 'O0_' in tmp:
                file_list_O0.append(tmp)
            elif 'O1_' in tmp:
                file_list_O1.append(tmp)
            elif 'O2_' in tmp:
                file_list_O2.append(tmp)
            elif 'O3_' in tmp:
                file_list_O3.append(tmp)
            else:
                file_list_Os.append(tmp)

    train_datas =[]
    valid_datas =[]
    test_datas =[]

    for dataset in [file_list_O0, file_list_O1, file_list_O2, file_list_O3, file_list_Os]:
        tmp_train, tmp_val, tmp_test = get_split_precent(dataset)
        train_datas.extend(tmp_train)
        valid_datas.extend(tmp_val)
        test_datas.extend(tmp_test)

    split_dict = {}
    split_dict["train"] = train_datas
    split_dict["valid"] = valid_datas
    split_dict["test"] = test_datas


    for split in ['train', 'valid', 'test']:
        list_path = outpath + os.sep + f"{split}_{arch}_list.txt"
        with open(list_path, 'w') as f:
            for line in split_dict[split]:
                f.write(line+'\n')


def generate_label_vocab(path, outpath, max_size=13000, min_freq=1):

    path_list =[]
    with open(path, 'r') as f:
        for path in f:
            path = path.strip('\n')

            data_path = os.path.dirname(path)
            tmp_name = os.path.basename(path).split('_')[:-2]
            ds_name = '_'.join(tmp_name)
            name = ds_name+'_cfg_graph_labels.txt'
            path_list.append(os.path.join(data_path, name))

    vocab = WordVocab(path_list, max_size=max_size, min_freq=min_freq)

    print("VOCAB SIZE:", len(vocab))
    # print(vocab.itos)
    vocab.save_vocab(outpath)

    return vocab

def load_k_hop_feature_data(datapre, features_pre,node_vocab, graph_label, label2id, node_num, node_len, target_len, feature_num, outpath ='test.pkl', arch='x86'):
    split_dict = {}

    for split in ['train', 'valid', 'test']:
        split_dict[f"{split}_data"]= []
        split_dict[f"{split}_label"] = []
        list_path = datapre + os.sep + f"{split}_{arch}_list.txt"

        with open(list_path, 'r') as f:
            for path in f:
                path = path.strip('\n')
                name = os.path.basename(path)

                data_path = os.path.dirname(path)
                tmp_name = name.split('_')[:-2]
                ds_name = '_'.join(tmp_name)

                feature_path = data_path.split('/')[-1]
                feature_path = features_pre + os.sep + feature_path

                temp_Gs, temp_raw_data, temp_labels, tmp_levels, tmp_fea_data = \
                    load_one_feature_data(data_path, feature_path, ds_name, 'cfg', node_vocab, graph_label, label2id, node_len, target_len, feature_num, node_num)
                Gs, raw_data, labels, levels, fea_data = process_feature_graph(temp_Gs, temp_raw_data, temp_labels, tmp_levels, tmp_fea_data, node_num, node_len)

                tmp_data = []
                for g, i, fea in zip(Gs, raw_data, fea_data):
                    tmp_data.append((g, i, fea))
                    if g.number_of_edges() == 0:
                            print(datapre, ds_name)

                tmp_tgt =[]
                for l1, l2 in zip(labels, levels):
                    tmp_tgt.append((l1, l2))

                split_dict[f"{split}_data"].extend(tmp_data)
                split_dict[f"{split}_label"].extend(tmp_tgt)


    save(split_dict, outpath)


def fill_list (my_list, length, fill=None):
    my_list = my_list.replace('[', '').replace(']', '').strip()
    my_list = my_list.split(',')
    my_list = [float(x) for x in my_list if len(x)>0]
    if len(my_list)> 0:
        max_path = int(max(my_list))
        out_dict = Counter(my_list)

        out_list = []

        for i in range(max_path+1):
            i = float(i)
            if  i in out_dict.keys():
                out_list.append(out_dict[i])
            else:
                out_list.append(0)
    else:
        out_list=[]

    if len(out_list) >=length:
        return out_list[:length]
    else:
        return out_list + (length - len(out_list)) * [fill]


def load_one_feature_data(datapre, feature_path, ds_name, datatype, node_vocab, graph_label, label2id, node_len, target_len, feature_num, node_num):
    node2graph = {}
    Gs = []
    with open("{}/{}_{}_graph_indicator.txt".format(datapre, ds_name, datatype), "r") as f:
        c = 1

        for line in f:

            line = line.strip()
            node2graph[c] = int(line)
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1

    with open("{}/{}_{}_A.txt".format(datapre, ds_name, datatype), "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            # edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])] - 1].add_edge(int(edge[0]), int(edge[1]))


    with open("{}/{}_{}_node_attributes.txt".format(datapre, ds_name, datatype), "r") as f:
            c = 1
            for line in f:
                node_label = line.strip()  # int(line[:-1])
                # l = (len(node_label.split(' '))) * [1]
                s = node_vocab.to_seq(node_label)
                s = [3] + s + [2]
                if len(s) > node_len:
                    Gs[node2graph[c] - 1].nodes[c]["label"] = s[:node_len]
                else:
                    Gs[node2graph[c] - 1].nodes[c]["label"] = s + [0] * (node_len - len(s))

                # Gs[node2graph[c]-1].node[c]['label'] = node_vocab.to_seq(node_label)
                c += 1

    labels = []
    tmp_lables = []
    with open("{}/{}_{}_graph_labels.txt".format(datapre, ds_name, datatype), "r") as f:
        for line in f:
            tmp_lables.append(line.strip())
            label = graph_label.to_seq(line, target_len)
            labels.append(label)

    levels = []
    with open("{}/{}_{}_level.txt".format(datapre, ds_name, datatype), "r") as f:
        for line in f:
            level = label2id[line.strip()]
            levels.append(level)
    # labels  = np.array(labels, dtype = np.float)

    raw_data = []
    with open("{}/{}_inst.txt".format(datapre, ds_name), "r") as f:

        for line in f:
            raw_inst = line.strip()  # int(line[:-1])
            raw_insts = raw_inst.split('\t')

            tmp_inst = []
            for node_label in raw_insts:

                s = node_vocab.to_seq(node_label)
                s = [3] + s + [2]
                inst_dict = {}
                if len(s) > node_len:
                    inst_dict["label"] = s[:node_len]
                else:
                    inst_dict["label"] = s + [0] * (node_len - len(s))

                tmp_inst.append(inst_dict)

            raw_data.append(tmp_inst)

    feature_data = dict()
    with codecs.open("{}/{}.csv".format(feature_path, ds_name)) as f:

        for line in csv.DictReader(f, delimiter='\t', skipinitialspace=True):
            value_data = []
            for key in attrValueInfo:
                value_data.append(float(line[key]))

            list_data = fill_list(line[attrListInfo[0]], feature_num - len(attrValueInfo), 0)
            str_list = get_proc_str_list(line[attrSetInfo[0]], node_vocab, node_num)
            feature_data[line['Func_Name']] = value_data + list_data + str_list

    final_fea_data = []

    for name in tmp_lables:
        final_fea_data.append(feature_data[name])

    assert  len(Gs) == len(labels) == len(levels)

    for G in Gs:
        if G.number_of_edges()==0:
            print(datapre, ds_name)

    return Gs, raw_data, labels, levels, final_fea_data

def process_feature_graph(Gs, raw_data, labels, levels, feature_data, node_num, node_len):
    node_labels = dict()
    out_Gs = []
    out_labels = []
    out_levels = []
    out_insts = []
    out_fea = []
    n_node = 1
    for G, inst, label, level, fea in zip(Gs, raw_data, labels, levels, feature_data):
        if G.number_of_nodes() > 512:
            n_node += G.number_of_nodes()
            continue
        if G.number_of_nodes() < 6:
            n_node += G.number_of_nodes()
            continue

        edges = np.asarray(nx.adjacency_matrix(G).todense())
        if G.number_of_nodes() > node_num:
            edges = edges[:node_num, :node_num]
            out_inst = inst[:node_num]
        else:
            delta_len = node_num - len(inst)
            out_inst = inst
            for _ in range(delta_len):
                tmp_inst = {}
                tmp_inst["label"] = [0] * node_len

                out_inst.append(tmp_inst)
        A_out = np.int32(edges)
        graph = nx.from_numpy_array(A_out)

        for i in graph.nodes():
            tmp_i = i + n_node
            if tmp_i in G.nodes():  # G.number_of_nodes():
                graph.nodes[i]["label"] = G.nodes[i + n_node]["label"]

        n_node += G.number_of_nodes()
        if graph.number_of_edges() == 0:
            print("error!")
        out_insts.append(out_inst)
        out_Gs.append(graph)
        out_labels.append(label)
        out_levels.append(level)
        out_fea.append(fea)

    return out_Gs, out_insts, out_labels, out_levels, out_fea

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datapre', type=str, default='', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='firmware', help='Dataset name')
parser.add_argument('--arch', type=str, default='X64_mix')

parser.add_argument('--feature_pre', type=str, default='',
                    help='location of the select feature data corpus')
parser.add_argument('--label_vocab_path', type=str, default='', help = 'path to save name vocab')
parser.add_argument('--inst_vocab_path', type=str, default='', help = 'path for pre-train language model generate vocab')
parser.add_argument('--max_vocab_size', type=int, required=False, default=13000)
parser.add_argument('--min_frequency', type=int, required=False, default=1)

parser.add_argument('--dataset_type', type=str, default='k-hop', help='the type is k-hop or gcn')
parser.add_argument('--max_label_num', type=int, default=10, help='function name length')
parser.add_argument('--node_len', type=int, default=20, required=False, help='function instruction length')
parser.add_argument('--node_num', type=int, default=512, required=False, help='function instruction size')
parser.add_argument('--feature_num', default=256, type=int, metavar='N', help='select feature num')
args = parser.parse_args()

if __name__ == '__main__':

    train_file_list = args.datapre + os.sep + f"train_{args.arch}_list.txt"
    if not os.path.exists(train_file_list):
        print("generating dataset split!!!!")
        get_dataset_split(args.datapre, args.datapre, args.arch)

    if not os.path.exists(args.label_vocab_path):
        print("Remove exist label vocab!")
        #os.remove(args.label_vocab_path)

        print("Generating Graph Lable Vocab", args.label_vocab_path)
        graph_label = generate_label_vocab(train_file_list, args.label_vocab_path, max_size=args.max_vocab_size, min_freq=args.min_frequency)
    else:
        graph_label = WordVocab.load_vocab(args.label_vocab_path)

    print("Loading Node Vocab", args.inst_vocab_path)
    node_vocab = dataset.WordVocab.load_vocab(args.inst_vocab_path)

    label2id = {'O0': 0, 'O1': 1, 'O2': 2, 'O3': 3, 'Os': 4}
    id2label = {0 :'O0', 1: 'O1', 2: 'O2', 3: 'O3', 4: 'Os'}

    dataset_path = '{}/{}_{}_{}.pkl'.format(args.datapre, args.dataset, args.arch, args.dataset_type)
    if os.path.exists(dataset_path):
        print("Remove exist dataset")
        os.remove(dataset_path)

    print("generate the dataset {}".format(dataset_path))

    load_k_hop_feature_data(args.datapre, args.feature_pre, node_vocab, graph_label, label2id, args.node_num, args.node_len,
                            args.max_label_num, args.feature_num, outpath=dataset_path, arch=args.arch)

    print("Generate dataset finished!")
