import os
import sys
import pickle
import numpy as np
import glob
import networkx as nx
import random
from graph_label_vocab import WordVocab
sys.path.append('../pre_train')
import dataset
from itertools import combinations
import hashlib


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

def get_dataset_split(datapre, outpath, arch='X64'):
    file_list = []
    for idx, datadir_name in enumerate(os.listdir(datapre)):
        tmp_path = os.path.join(datapre, datadir_name)
        temp_filters = glob.glob(tmp_path + os.sep + "*_cfg_A.txt")
        file_list.extend(temp_filters)

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

    split_dict = {}
    split_dict["train"] = file_list[:offset0]
    split_dict["valid"] = file_list[offset0:offset0 + offset1]
    split_dict["test"] = file_list[offset0 + offset1:]


    for split in ['train', 'valid', 'test']:
        list_path = outpath + os.sep + f"{split}_{arch}_list.txt"
        with open(list_path, 'w') as f:
            for line in split_dict[split]:
                f.write(line+'\n')


def generate_label_vocab(inpaths, outpath, max_size=13000, min_freq=1):

    path_list =[]
    for inpath in inpaths:
        with open(inpath, 'r') as f:
            for path in f:
                path = path.strip('\n')

                data_path = os.path.dirname(path)

                tmp_name = os.path.basename(path).split('_')[:-2]
                ds_name = '_'.join(tmp_name)
                name = ds_name+'_cfg_graph_labels.txt'
                for i in ['O0', 'O1','O2','O3', 'Os']:
                    path_list.append(os.path.join( data_path.replace('O0',i), name))

    vocab = WordVocab(path_list, max_size=max_size, min_freq=min_freq)

    print("VOCAB SIZE:", len(vocab))

    vocab.save_vocab(outpath)

    return vocab


def generate_positive(key, value):
    output = list(combinations(value, 2))

    if len(value) > 3:
        true_value = random.sample(output, 5)
        label = len(true_value) * [1]
    else:
        true_value_v1 = random.sample(output, 5-len(output))
        true_value = output + true_value_v1

        label = len(true_value) * [1]
    
    return true_value, label


def list2str(data_list):
    out = []

    for insts in data_list:
        for i in insts:
            out.append(str(i))

    return ' '.join(out)


def generate_muli_opt_data(data_dict, class_dict = None):
    og = data_dict['O0']
    o1 = data_dict['O1']
    o2 = data_dict['O2']
    o3 = data_dict['O3']
    Os = data_dict['Os']
    final_data =[]
    final_class = []

    keys_list = set(list(og.keys())+list(o1.keys()) + list(o2.keys()) + list(o3.keys()) + list(Os.keys()))
    for key in keys_list:
        good_value =[]
        hash_list = set()
        try:
            _,inst,_ = og[key]
            inst = list2str(inst)
            hash_code = hashlib.md5(inst.encode(encoding='utf-8')).hexdigest()
            if hash_code not in hash_list:
                hash_list.add(hash_code)
                good_value.append(og[key])

        except:
            pass

        try:
            _, inst, _ = o1[key]
            inst = list2str(inst)
            hash_code = hashlib.md5(inst.encode(encoding='utf-8')).hexdigest()
            if hash_code not in hash_list:
                hash_list.add(hash_code)
                good_value.append(o1[key])
        except:
            pass

        try:
            _, inst, _ = o2[key]
            inst = list2str(inst)
            hash_code = hashlib.md5(inst.encode(encoding='utf-8')).hexdigest()
            if hash_code not in hash_list:
                hash_list.add(hash_code)
                good_value.append(o2[key])
        except:
            pass

        try:
            _, inst, _ = o3[key]
            inst = list2str(inst)
            hash_code = hashlib.md5(inst.encode(encoding='utf-8')).hexdigest()
            if hash_code not in hash_list:
                hash_list.add(hash_code)
                good_value.append(o3[key])
        except:
            pass

        try:
            _, inst, _ = Os[key]
            inst = list2str(inst)
            hash_code = hashlib.md5(inst.encode(encoding='utf-8')).hexdigest()
            if hash_code not in hash_list:
                hash_list.add(hash_code)
                good_value.append(Os[key])
        except:
            pass

        if len(good_value) > 2:
            tmp_data, _ = generate_positive(key, good_value)
            final_data.extend(tmp_data)
            if class_dict is not None:
                class_ = class_dict[key]
                final_class.extend([class_]* len(tmp_data))
        else:
            if len(good_value)==2:
                single_data = [(good_value[0], good_value[1])] * 5
            else:
                single_data = [(good_value[0], good_value[0])] * 5
            final_data.extend(single_data)
            if class_dict is not None:
                class_ = class_dict[key]
                final_class.extend([class_] * 5)


    if class_dict is not None:
        return  final_data, final_class
    else:
        return final_data



def load_k_hop_data(datapre, node_vocab, graph_label, node_num, node_len, target_len, outpath ='test.pkl', arch='X64', small=False):
    split_dict = {}
    class_dict = {}
    opt_dict = {}

    list_path = datapre + os.sep + f"train_{arch}_list.txt"

    split_dict["triple_data"] = []
    split_dict["triple_label"] = []
    split_dict["triple_class"] = []

    with open(list_path, 'r') as f:
        diver_data = []
        diver_class =[]
        path_list = f.readlines()
        if small:
            path_len = int(len(path_list) * 0.5)
            path_list = path_list[:path_len]

        for path in path_list:
            tmp_diver_data = {}
            path = path.strip('\n')
            name = os.path.basename(path)
            data_path = os.path.dirname(path)

            tmp_name = name.split('_')[:-2]
            ds_name = '_'.join(tmp_name)

            for opt in ['O0', 'O1','O2','O3', 'Os']:
                temp_data = dict()
                temp_Gs, temp_raw_data, temp_labels = load_one_data(data_path.replace('O0', opt), ds_name, 'cfg', node_vocab,
                                                                                graph_label, node_len, target_len, use_train=True)
                Gs, raw_data, labels = process_graph(temp_Gs, temp_raw_data, temp_labels, node_num,
                                                             node_len)

                for g, inst, label in zip(Gs, raw_data, labels):
                    temp_data[label] = (g, inst, label)
                    if label not in class_dict:
                        class_dict[label] = len(class_dict)+1

                tmp_diver_data[f'{opt}'] = temp_data

            tmp_data, temp_class = generate_muli_opt_data(tmp_diver_data, class_dict=class_dict)

            diver_data.extend(tmp_data)
            diver_class.extend(temp_class)

    for funcs, class_ in zip(diver_data, diver_class) :

        _, _, l1 = funcs[0]
        l1 = graph_label.to_seq(l1, target_len)

        split_dict["triple_data"].append(funcs)
        split_dict["triple_label"].append(l1)
        split_dict["triple_class"].append(class_)

    for split in ['valid']:
        split_dict[f"{split}_data"]= []
        split_dict[f"{split}_label"] = []
        list_path = datapre + os.sep + f"{split}_{arch}_list.txt"

        with open(list_path, 'r') as f:
            path_list = f.readlines()
            if small:
                path_len = int(len(path_list)* 0.5)
                path_list = path_list[:path_len]

            for path in path_list:
                path = path.strip('\n')
                name = os.path.basename(path)

                data_path = os.path.dirname(path)

                tmp_name = name.split('_')[:-2]
                ds_name = '_'.join(tmp_name)

                data_path = data_path.replace('O0', 'O2')

                temp_Gs, temp_raw_data, temp_labels = load_one_data(data_path, ds_name, 'cfg', node_vocab, graph_label,node_len, target_len)
                Gs, raw_data, labels= process_graph(temp_Gs, temp_raw_data, temp_labels, node_num, node_len)

                tmp_data = []
                for g, i in zip(Gs, raw_data):
                    tmp_data.append((g, i))
                    if g.number_of_edges() == 0:
                            print(datapre, ds_name)

                split_dict[f"{split}_data"].extend(tmp_data)
                split_dict[f"{split}_label"].extend(labels)


    for opt in ['O0', 'O1', 'O2', 'O3', 'Os']:
        opt_dict[f"{opt}_data"] =[]
        opt_dict[f"{opt}_label"] = []
        list_path = datapre + os.sep + f"test_{arch}_list.txt"

        with open(list_path, 'r') as f:
            path_list = f.readlines()
            if small:
                path_len = int(len(path_list)* 0.5)
                path_list = path_list[:path_len]

            for path in path_list:
                path = path.strip('\n')
                name = os.path.basename(path)

                data_path = os.path.dirname(path)

                tmp_name = name.split('_')[:-2]
                ds_name = '_'.join(tmp_name)

                data_path = data_path.replace('O0', opt)

                temp_Gs, temp_raw_data, temp_labels = load_one_data(data_path, ds_name, 'cfg', node_vocab, graph_label,node_len, target_len)
                Gs, raw_data, labels= process_graph(temp_Gs, temp_raw_data, temp_labels, node_num, node_len)

                tmp_data = []
                for g, i in zip(Gs, raw_data):
                    tmp_data.append((g, i))
                    if g.number_of_edges() == 0:
                            print(datapre, ds_name)

                opt_dict[f"{opt}_data"].extend(tmp_data)
                opt_dict[f"{opt}_label"].extend(labels)


    save(split_dict, outpath)

    test_dirname = os.path.dirname(outpath)
    test_name = os.path.basename(outpath)
    test_name = 'test_' + test_name
    test_out = os.path.join(test_dirname, test_name)
    save(opt_dict, test_out)



def load_one_data(datapre, ds_name, datatype, node_vocab, graph_label, node_len, target_len, use_train=False):
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

            Gs[node2graph[int(edge[0])] - 1].add_edge(int(edge[0]), int(edge[1]))


    with open("{}/{}_{}_node_attributes.txt".format(datapre, ds_name, datatype), "r") as f:
            c = 1
            for line in f:
                node_label = line.strip()  # int(line[:-1])

                s = node_vocab.to_seq(node_label)
                s = [node_vocab.sos_index] + s + [node_vocab.eos_index]

                if len(s) > node_len:
                    Gs[node2graph[c] - 1].nodes[c]["label"] = s[:node_len]
                else:
                    Gs[node2graph[c] - 1].nodes[c]["label"] = s + [node_vocab.pad_index] * (node_len - len(s))


                c += 1

    labels = []

    with open("{}/{}_{}_graph_labels.txt".format(datapre, ds_name, datatype), "r") as f:
        for line in f:
            if use_train:
                label =line.strip()
                labels.append(label)
            else:
                label = graph_label.to_seq(line, target_len)
                labels.append(label)


    raw_data = []
    with open("{}/{}_inst.txt".format(datapre, ds_name), "r") as f:

        for line in f:
            raw_inst = line.strip()  # int(line[:-1])
            raw_insts = raw_inst.split('\t')

            tmp_inst = []
            for node_label in raw_insts:

                s = node_vocab.to_seq(node_label)
                s = [node_vocab.sos_index] + s + [node_vocab.eos_index]
                if len(s) > node_len:
                    inst_dict = s[:node_len]
                else:
                    inst_dict = s + [node_vocab.pad_index] * (node_len - len(s))

                tmp_inst.append(inst_dict)

            raw_data.append(tmp_inst)

    assert  len(Gs) == len(labels)

    for G in Gs:
        if G.number_of_edges()==0:
            print(datapre, ds_name)


    return Gs, raw_data, labels



def process_graph(Gs, raw_data, labels, node_num, node_len):

    out_Gs = []
    out_labels = []
    out_insts = []
    n_node = 1
    for G, inst, label in zip(Gs, raw_data, labels):
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
                tmp_inst = [0] * node_len

                out_inst.append(tmp_inst)

        A_out = np.int32(edges)
        graph = nx.from_numpy_array(A_out)
        # print('graph:', graph.nodes())
        # print('G:', G.nodes())
        # IPython.embed()

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


    return out_Gs, out_insts, out_labels



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datapre', type=str, default='', help='location of the data corpus')
parser.add_argument('--arch', type=str, default='X64')
parser.add_argument('--label_vocab_path', type=str, default='', help='the path to save target vocab')
parser.add_argument('--inst_vocab_path', type=str, default='../pretrain/modelout/vocab_X64')
parser.add_argument('--max_vocab_size', type=int, required=False, default=10000)
parser.add_argument('--min_frequency', type=int, required=False, default=1)

parser.add_argument('--max_label_num', type=int, default=10, help='function name length')
parser.add_argument('--node_len', type=int, default=16, required=False, help='function name length')
parser.add_argument('--node_num', type=int, default=256, required=False, help='function name length')
parser.add_argument('--feature_num', default=256, type=int, metavar='N', help='select feature num')
args = parser.parse_args()

if __name__ == '__main__':

    train_file_list = args.datapre + os.sep + f"train_{args.arch}_list.txt"
    if not os.path.exists(train_file_list):
        print("generating dataset split!!!!")
        get_dataset_split(args.datapre, args.datapre, args.arch)

    if not os.path.exists(args.label_vocab_path):
        print("Remove exist label vocab!")
        os.remove(args.label_vocab_path)

        print("Generating Graph Lable Vocab", args.label_vocab_path)
        file_list = [train_file_list]
        graph_label = generate_label_vocab(file_list, args.label_vocab_path, max_size=args.max_vocab_size, min_freq=args.min_frequency)
    else:
        graph_label = WordVocab.load_vocab(args.label_vocab_path)

    print("Loading Node Vocab", args.inst_vocab_path)
    node_vocab = dataset.WordVocab.load_vocab(args.inst_vocab_path)

    dataset_path = args.datapre + '/X64_dataset.pkl'

    if os.path.exists(dataset_path):
        print("Remove exist dataset")
        #os.remove(dataset_path)

    print("generate the dataset {}".format(dataset_path))
    load_k_hop_data(args.datapre, node_vocab, graph_label, args.node_num, args.node_len, args.max_label_num,
                    outpath = dataset_path, arch=args.arch, small=False)

    print("Generate dataset finished!")

