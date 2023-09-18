import networkx as nx
import numpy as np
import scipy.sparse as sp
import random
import torch
import functools
import torch.utils.data as data
from sklearn.utils import shuffle
from model_config import Config

def set_seed(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Mydataset(data.Dataset):
    def __init__(self, x, y, class_=None, is_train=False):
        self.data = x
        self.y = y
        self.class_ = class_
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            temp_datas = self.data[index]
            inputs = []
            for tmp_data in temp_datas:
                Gs, inst, _ = tmp_data
                inputs.append((Gs, inst))
            target = self.y[index]
            class_ = self.class_[index]

            return inputs, (target, class_)

        else:
            input_Gs, input_inst = self.data[index]
            target = self.y[index]
            return [input_Gs, input_inst], target

    def __len__(self):
        return len(self.y)


def preprocess_adj(edges_batch, features_batch, seg_batch):

    adj_batch = list()
    for i in range(2 * Config.radius):
        if i % 2 == 0:
            edges = np.vstack(edges_batch[i])
            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(len(features_batch[i // 2]), len(features_batch[i // 2 + 1])), dtype=np.float32)
        else:
            if len(edges_batch[i]) == 0:
                adj = sp.coo_matrix((len(features_batch[i // 2 + 1]), len(features_batch[i // 2 + 1])),
                                    dtype=np.float32)
            else:
                edges = np.vstack(edges_batch[i])
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                    shape=(len(features_batch[i // 2 + 1]), len(features_batch[i // 2 + 1])),
                                    dtype=np.float32)

        adj = sparse_mx_to_torch_sparse_tensor(adj)
        adj_batch.append(adj.cuda())

    for i in range(Config.radius + 1):
        features_batch[i] = torch.LongTensor(np.asarray(features_batch[i])).cuda()
        seg_batch[i] = torch.IntTensor(np.asarray(seg_batch[i])).cuda()

    return adj_batch, features_batch, seg_batch

def train_triple_collate_fn(batch):

    anchor =[]
    targets =[]

    for inputs, target in batch:
        for i in inputs:
            anchor.append(i)
            targets.append(target)

    anchor, targets = shuffle(anchor, targets, random_state=40)

    bs = len(anchor)
    radius = Config.radius
    target_len = Config.target_len

    n_nodes = 0
    for inputs in anchor:
        Gs, _ = inputs
        n_nodes += Gs.number_of_nodes()

    idx_batch = np.zeros(bs + 1, dtype=np.int64)
    idx_batch[0] = 0
    idx_node = np.zeros(n_nodes, dtype=np.int64)

    class_batch = np.zeros((bs), dtype=np.int32)
    y_batch = np.zeros((bs, target_len))

    edges_batch = list()
    for _ in range(radius * 2):
        edges_batch.append(list())

    tuple_to_idx = list()
    features_batch = list()
    seg_batch = list()
    inst_batch = list()
    inst_seg_batch = list()
    for _ in range(radius + 1):
        tuple_to_idx.append(dict())
        features_batch.append(list())
        seg_batch.append(list())

    for j, ((Gs, insts), (label, class_)) in enumerate(zip(anchor, targets)):
        n = Gs.number_of_nodes()
        feat = dict()
        seg = dict()

        for node in Gs.nodes():
            feat[node] = Gs.nodes[node]["label"]
            seg[node] = np.greater(feat[node], 0).astype(np.int32)

        for k, n1 in enumerate(Gs.nodes()):
            idx_node[idx_batch[j] + k] = j

            ego = nx.ego_graph(Gs, n1, radius=radius)
            dists = nx.single_source_shortest_path_length(ego, n1)

            for n2 in ego.nodes():
                tuple_to_idx[dists[n2]][(n1, n2)] = len(tuple_to_idx[dists[n2]])
                features_batch[dists[n2]].append(feat[n2])
                seg_batch[dists[n2]].append(seg[n2])

            for n2 in ego.nodes():
                for n3 in ego.neighbors(n2):
                    if dists[n3] > dists[n2]:
                        edges_batch[2 * dists[n2]].append(
                            (tuple_to_idx[dists[n2]][(n1, n2)], tuple_to_idx[dists[n2] + 1][(n1, n3)]))
                    elif dists[n3] == dists[n2]:
                        edges_batch[2 * dists[n2] - 1].append(
                            (tuple_to_idx[dists[n2]][(n1, n2)], tuple_to_idx[dists[n2]][(n1, n3)]))

        inst_feat = []
        inst_seg = []

        for inst in insts:
            inst_feat.append(np.asarray(inst))
            inst_seg.append(np.greater(inst, 0).astype(np.int32))

        idx_batch[j + 1] = idx_batch[j] + n
        inst_batch.append(np.asarray(inst_feat))
        inst_seg_batch.append(np.asarray(inst_seg))
        y_batch[j, :] = label
        class_batch[j] = class_


    if Config.multi_gpu:
        inst_batch = torch.LongTensor(np.asarray(inst_batch))
        inst_seg_batch = torch.IntTensor(np.asarray(inst_seg_batch))
        Gs_tensors = (inst_batch, inst_seg_batch, edges_batch, features_batch, seg_batch, torch.LongTensor(idx_node))
        targets = (torch.LongTensor(y_batch), torch.IntTensor(class_batch))

    else:
        adj_batch = list()
        for i in range(2 * radius):

            if i % 2 == 0:
                edges = np.vstack(edges_batch[i])
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                    shape=(len(features_batch[i // 2]), len(features_batch[i // 2 + 1])),
                                    dtype=np.float32)
            else:
                if len(edges_batch[i]) == 0:
                    adj = sp.coo_matrix((len(features_batch[i // 2 + 1]), len(features_batch[i // 2 + 1])),
                                        dtype=np.float32)
                else:
                    edges = np.vstack(edges_batch[i])
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(len(features_batch[i // 2 + 1]), len(features_batch[i // 2 + 1])),
                                        dtype=np.float32)

            adj_batch.append(sparse_mx_to_torch_sparse_tensor(adj).cuda())

        for i in range(radius + 1):
            features_batch[i] = torch.LongTensor(np.asarray(features_batch[i])).cuda()
            seg_batch[i] = torch.IntTensor(np.asarray(seg_batch[i])).cuda()

        inst_batch = torch.LongTensor(np.asarray(inst_batch)).cuda()
        inst_seg_batch = torch.IntTensor(np.asarray(inst_seg_batch)).cuda()

        Gs_tensors = (inst_batch, inst_seg_batch, adj_batch, features_batch, seg_batch, torch.LongTensor(idx_node).cuda())
        targets = (torch.LongTensor(y_batch).cuda(), torch.IntTensor(class_batch).cuda())

    return Gs_tensors, targets


def collate_fn(batch):

    bs = len(batch)
    radius = Config.radius
    target_len = Config.target_len

    n_nodes = 0
    for inputs, _ in batch:
        Gs, _ = inputs
        n_nodes += Gs.number_of_nodes()

    y_batch = np.zeros((bs, target_len))

    idx_batch = np.zeros(bs + 1, dtype=np.int64)
    idx_batch[0] = 0
    idx_node = np.zeros(n_nodes, dtype=np.int64)

    edges_batch = list()
    for _ in range(radius * 2):
        edges_batch.append(list())

    tuple_to_idx = list()
    features_batch = list()
    seg_batch = list()
    inst_batch = list()
    inst_seg_batch = list()
    for _ in range(radius + 1):
        tuple_to_idx.append(dict())
        features_batch.append(list())
        seg_batch.append(list())

    for j, ([Gs, insts], label) in enumerate(batch):
        n = Gs.number_of_nodes()
        feat = dict()
        seg = dict()

        for node in Gs.nodes():
            feat[node] = Gs.nodes[node]["label"]
            seg[node] = np.greater(feat[node], 0).astype(np.int32)

        for k, n1 in enumerate(Gs.nodes()):
            idx_node[idx_batch[j] + k] = j

            ego = nx.ego_graph(Gs, n1, radius=radius)
            dists = nx.single_source_shortest_path_length(ego, n1)

            for n2 in ego.nodes():
                tuple_to_idx[dists[n2]][(n1, n2)] = len(tuple_to_idx[dists[n2]])
                features_batch[dists[n2]].append(feat[n2])
                seg_batch[dists[n2]].append(seg[n2])


            for n2 in ego.nodes():
                for n3 in ego.neighbors(n2):
                    if dists[n3] > dists[n2]:
                        edges_batch[2 * dists[n2]].append(
                            (tuple_to_idx[dists[n2]][(n1, n2)], tuple_to_idx[dists[n2] + 1][(n1, n3)]))
                    elif dists[n3] == dists[n2]:
                        edges_batch[2 * dists[n2] - 1].append(
                            (tuple_to_idx[dists[n2]][(n1, n2)], tuple_to_idx[dists[n2]][(n1, n3)]))

        inst_feat = []
        inst_seg = []

        for inst in insts:
            inst_feat.append(np.asarray(inst))
            inst_seg.append(np.greater(inst, 0).astype(np.int32))

        idx_batch[j + 1] = idx_batch[j] + n
        y_batch[j, :] = label
        inst_batch.append(np.asarray(inst_feat))
        inst_seg_batch.append(np.asarray(inst_seg))

    if Config.multi_gpu:
        inst_batch = torch.LongTensor(np.asarray(inst_batch))
        inst_seg_batch = torch.IntTensor(np.asarray(inst_seg_batch))

        Gs_tensors = (inst_batch, inst_seg_batch, edges_batch, features_batch, seg_batch, torch.LongTensor(idx_node))
        targets = torch.LongTensor(y_batch)
        return Gs_tensors, targets

    else:
        adj_batch = list()

        for i in range(2 * radius):

            if i % 2 == 0:
                edges = np.vstack(edges_batch[i])
                adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                    shape=(len(features_batch[i // 2]), len(features_batch[i // 2 + 1])),
                                    dtype=np.float32)
            else:
                if len(edges_batch[i]) == 0:
                    adj = sp.coo_matrix((len(features_batch[i // 2 + 1]), len(features_batch[i // 2 + 1])),
                                        dtype=np.float32)
                else:
                    edges = np.vstack(edges_batch[i])
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(len(features_batch[i // 2 + 1]), len(features_batch[i // 2 + 1])),
                                        dtype=np.float32)

            adj_batch.append(sparse_mx_to_torch_sparse_tensor(adj).cuda())

        for i in range(radius + 1):
            features_batch[i] = torch.LongTensor(np.asarray(features_batch[i])).cuda()
            seg_batch[i] = torch.IntTensor(np.asarray(seg_batch[i])).cuda()

        inst_batch = torch.LongTensor(np.asarray(inst_batch)).cuda()
        inst_seg_batch = torch.IntTensor(np.asarray(inst_seg_batch)).cuda()

        Gs_tensors = (
        inst_batch, inst_seg_batch, adj_batch, features_batch, seg_batch, torch.LongTensor(idx_node).cuda())
        targets = torch.LongTensor(y_batch).cuda()

        return Gs_tensors, targets


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_performance(vocab, preds, gold):

    with open(Config.word_net_path, 'r') as f:
        word_cluster = json.load(f)

    true_positive, false_positive, false_negative = 0, 0, 0
    total = 0

    special_tokens = [vocab.pad_index, vocab.sos_index, vocab.eos_index]
    for ref, pred in zip(gold, preds):
        total += 1
        target=[]
        prediction=[]
        for p in pred:
            if p in special_tokens:
                continue
            prediction.append(vocab.itos[p])
        for t in ref:
            if t in special_tokens:
                continue
            target.append(vocab.itos[t])

        tp, fp, fn = get_correct_predictions_word_cluster(target, prediction, word_cluster)
        true_positive += tp
        false_positive += fp
        false_negative += fn

    return true_positive, false_positive, false_negative



def get_correct_predictions_word_cluster(target, prediction, word_cluster):
    """
    Calculate predictions based on word cluster generated by CodeWordNet.
    """

    true_positive, false_positive, false_negative = 0, 0, 0
    replacement = dict()
    skip = set()
    for j, p in enumerate(prediction):
        if p in target:
            skip.add(j)
    for i, t in enumerate(target):
        for j, p in enumerate(prediction):
            if t != p and j not in replacement and j not in skip:

                if t in word_cluster and p in word_cluster:
                    t_cluster = word_cluster[t]
                    p_cluster = word_cluster[p]
                    t_cluster, p_cluster = set(t_cluster), set(p_cluster)
                    if len(t_cluster.intersection(p_cluster)) > 0:
                        replacement[j] = t

    for k, v in replacement.items():
        prediction[k] = v
    if target == prediction:
        true_positive = len(target)
    else:
        target = set(target)
        prediction = set(prediction)

        true_positive += len(target.intersection(prediction))
        false_negative += len(target.difference(prediction))
        false_positive += len(prediction.difference(target))
    return true_positive, false_positive, false_negative


def calculate_results(true_positive, false_positive, false_negative):
    # avoid dev by 0
    if true_positive + false_positive == 0:
        return 0, 0, 0
    if true_positive + false_negative == 0:
        return 0, 0, 0
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')
