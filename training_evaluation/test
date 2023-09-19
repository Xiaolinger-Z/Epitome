import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import datetime
import numpy as np
import time
import pickle
import gc
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
from graph_label_vocab import WordVocab
import sys
sys.path.append('../pre_train/pre_train_model')
import dataset
from model import Model
from model_util import AverageMeter, get_logger, set_seed, collate_fn, Mydataset, calculate_results, cal_performance, train_triple_collate_fn
from model_config import Config

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret



logfile = 'x64_dataset.txt'
logging = get_logger(log_path=os.path.join(Config.save_path, logfile))


class Run(object):
    def __init__(self):

        set_seed(Config.seed)
        out_name = 'X64_dataset'
        self.start_epoch = -1

        log_str = "device: {}, datapre: {}, lr :{}, dropout:{}, target_len: {}, node_num:{}, node_len:{}, batch-size:{}, emb_dim:{}, hidden_dim:{}". \
            format(Config.device, Config.datapre, Config.lr, Config.dropout, Config.target_len, Config.node_num,
                   Config.node_len,
                   Config.batch_size, Config.emb_dim, Config.hidden_dim)
        logging(log_str)

        print("Loading Node Vocab", Config.node_vocab_path)
        self.node_vocab = dataset.WordVocab.load_vocab(Config.node_vocab_path)

        print("Loading Graph Lable Vocab", Config.graph_label_vocab_path)
        self.graph_label = WordVocab.load_vocab(Config.graph_label_vocab_path)
        self.start_label = self.graph_label.stoi.get('<sos>')

        self.model = Model(Config, self.node_vocab, self.graph_label).cuda()

        self.checkpoint_best_path = Config.save_path + f'result/model_best_{out_name}_checkpoint.pth.tar'
        self.checkpoint_latest_path = Config.save_path + f'result/model_latest_{out_name}_checkpoint.pth.tar'

        logging(self.checkpoint_latest_path)


    def test_step(self, inputs, targets):

        bs, _ = targets.size()
        preds_t = np.zeros((bs, Config.target_len), np.int32)
        preds_t[:, 0] = self.start_label
        preds_t = torch.LongTensor(preds_t).cuda()

        preds = Variable(preds_t)
        for de_index in range(1, Config.target_len):
            _preds = self.model.evaluate(inputs, preds)
            preds_t[:, de_index] = _preds.data[:, de_index - 1]
            preds = Variable(preds_t.long())


        preds = preds.data.cpu().numpy()
        true_positive, false_positive, false_negative = cal_performance(self.graph_label, preds, targets)

        return preds, true_positive, false_positive, false_negative

    def test_beam_search_step(self, inputs, targets):


        bs, _ = targets.size()
        preds_t = np.zeros((bs, Config.target_len), np.int32)
        preds_t[:, 0] = self.start_label  # temp_satrt
        preds_t = torch.LongTensor(preds_t).cuda()

        preds = Variable(preds_t)

        preds = self.model.evaluate(inputs, preds)

        true_positive, false_positive, false_negative = cal_performance(self.graph_label, preds, targets)

        return preds, true_positive, false_positive, false_negative

    def test(self):

        Config.multi_gpu = False
        print("Loading checkpoint!")

        logging(self.checkpoint_best_path)
        checkpoint = torch.load(self.checkpoint_best_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        seqdata = pickle.load(open('{}.pkl'.format(Config.test_datapre), 'rb'))

        for opt in Config.test_opt:
            test_set = Mydataset(seqdata[f'{opt}_data'], seqdata[f'{opt}_label'])

            test_loader = DataLoader(test_set, batch_size=Config.batch_size, collate_fn=collate_fn, num_workers=0,
                                     drop_last=True)

            outfn = f'{Config.save_path}/predictions/predict_X64_{opt}.txt'
            outf = open(outfn, 'w')
            print("writing to file: " + outfn)

            true_positive, false_positive, false_negative = 0, 0, 0
            if Config.beam > 0:
                with torch.no_grad():
                    for Gs_test, y_test in test_loader:
                        preds, tp, fp, fn = self.test_beam_search_step(Gs_test, y_test)

                        true_positive += tp
                        false_positive += fp
                        false_negative += fn

                        for target, pred in zip(y_test, preds):  # sentence-wise
                            got = " ".join(
                                fil(self.graph_label.itos[idx] for idx in pred))  # .split("<eos>")[0].strip()
                            out_target = " ".join(fil(self.graph_label.itos[idx] for idx in target))
                            outf.write("- expected: " + out_target + "\n")
                            outf.write("- got: " + got + "\n\n")
                            outf.flush()
            else:

                with torch.no_grad():
                    for Gs_test, y_test in test_loader:
                        preds, tp, fp, fn = self.test_step(Gs_test, y_test)

                        true_positive += tp
                        false_positive += fp
                        false_negative += fn

                        for target, pred in zip(y_test, preds):  # sentence-wise
                            got = " ".join(
                                fil(self.graph_label.itos[idx] for idx in pred))  # .split("<eos>")[0].strip()
                            out_target = " ".join(fil(self.graph_label.itos[idx] for idx in target))
                            outf.write("- expected: " + out_target + "\n")
                            outf.write("- got: " + got + "\n\n")
                            outf.flush()

            outf.close()
            precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
            logging("opt {} dataset, Precision: {}, Recall: {}, F1: {}".format(opt, precision, recall, f1))


if __name__ == '__main__':
    # main()
    run = Run()
    run.test()
