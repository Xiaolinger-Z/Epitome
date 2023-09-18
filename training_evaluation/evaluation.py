import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import numpy as np
import pickle
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from graph_label_vocab import WordVocab
import sys
from sklearn.metrics import accuracy_score

sys.path.append('../pre_train/pre_train_model')
# sys.path.append('/root/CompilerOptimization/bart_func_name')
import dataset
from model import Model
from model_util import AverageMeter, collate_fn, Mydataset, cal_performance_sym, calculate_results
from model_config import ModelConfig


def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret


def main():

    work_dir = './modelout/'
    print("Loading Node Vocab", ModelConfig.node_vocab_path)
    node_vocab = dataset.WordVocab.load_vocab(ModelConfig.node_vocab_path)
    enc_vocab_size = len(node_vocab)

    print("Loading Graph Lable Vocab", ModelConfig.graph_label_vocab_path)
    graph_label = WordVocab.load_vocab(ModelConfig.graph_label_vocab_path)
    start_label = graph_label.stoi.get('<sos>')
    dec_vocab_size = len(graph_label)

    label2id = {'O0': 0, 'O1': 1, 'O2': 2, 'O3': 3, 'Os': 4}
    id2label = {0: 'O0', 1: 'O1', 2: 'O2', 3: 'O3', 4: 'Os'}

    seqdata = pickle.load(open('{}/firmware_X64_mix_k-hop_fea_512_18.pkl'.format(ModelConfig.datapre), 'rb'))

    print("load dataset finished!!!!")

    test_set = Mydataset(seqdata['test_data'], seqdata['test_label'])

    test_loader = DataLoader(test_set, batch_size=ModelConfig.batch_size, collate_fn=collate_fn, num_workers=0)

    model = Model(ModelConfig, enc_vocab_size, dec_vocab_size, len(label2id)).cuda()

    checkpoint_best_path = work_dir + f'result/model_best_{ModelConfig.dropout}_{ModelConfig.lr}_bsz_{ModelConfig.batch_size}_{ModelConfig.hidden_dim}.pth.tar'


    def test(inst, inst_seg, adj, features, segment, idx, select_fea, y, labels):

        bs, _ = y.size()
        preds_t = np.zeros((bs, ModelConfig.target_len), np.int32)
        preds_t[:, 0] = start_label  # temp_satrt
        preds_t = torch.LongTensor(preds_t).cuda()

        preds = Variable(preds_t)
        for de_index in range(1, ModelConfig.target_len):
            _preds = model.evaluate(inst, inst_seg, adj, features, segment, idx, select_fea, preds)
            preds_t[:, de_index] = _preds.data[:, de_index - 1]
            preds = Variable(preds_t.long())

        level_preds = model.evaluate_level(inst, inst_seg, adj, features, segment, idx, select_fea)
        level_acc = accuracy_score(labels.data.cpu().numpy().reshape(-1), level_preds.data.cpu().numpy().reshape(-1),
                                   normalize=True)

        true_positive, false_positive, false_negative = cal_performance_sym(graph_label, preds, y, 3)
        return preds, level_acc, true_positive, false_positive, false_negative


    # Testing
    test_acc = AverageMeter()
    test_level_acc = AverageMeter()

    print("Loading checkpoint!")
    checkpoint = torch.load(checkpoint_best_path)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # ("epoch:", epoch)
    outfn = work_dir + "predictions/" + str(ModelConfig.dropout) + '_' + str(ModelConfig.lr) + "_predict.txt"
    outf = open(outfn, 'w')
    print("writing to file: " + outfn)

    list_of_refs, hypotheses = [], []

    true_positive, false_positive, false_negative = 0, 0, 0
    with torch.no_grad():
        for Gs_test, y_test in test_loader:
                inst_test, inst_seg_test, adj_test, features_test, segment_test, idx_test, select_fea_test = Gs_test
                y_test, level = y_test
                preds, level_acc, tp, fp, fn = test(inst_test, inst_seg_test, adj_test, features_test, segment_test,
                                                    idx_test,
                                                    select_fea_test, y_test, level)
                test_tmp_acc = tp / (tp + fp)
                test_acc.update(test_tmp_acc, y_test.size(0))
                test_level_acc.update(level_acc, y_test.size(0))

                true_positive += tp
                false_positive += fp
                false_negative += fn
                # n_word_total += n_word
                # n_word_correct += n_correct

                for target, pred in zip(y_test, preds):  # sentence-wise
                    got = " ".join(fil(graph_label.itos[idx] for idx in pred))  # .split("<eos>")[0].strip()
                    out_target = " ".join(fil(graph_label.itos[idx] for idx in target))
                    outf.write("- expected: " + out_target + "\n")
                    outf.write("- got: " + got + "\n\n")
                    outf.flush()

                    # bleu score
                    ref = out_target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)

    precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)

    print("test_acc= {:.5f}, test_level_acc= {:.5f}".format(test_acc.avg, test_level_acc.avg))  # test_acc.avg
    print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))


if __name__ == '__main__':
    main()
