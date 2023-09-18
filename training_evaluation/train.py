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

        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        self.optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': 0.001}, {'params': bias_p, 'weight_decay': 0}],
            lr=Config.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=Config.factor,
                                                              patience=Config.patience)

        self.checkpoint_best_path = Config.save_path + f'result/model_best_{out_name}.pth.tar'
        self.checkpoint_latest_path = Config.save_path + f'result/model_latest_{out_name}.pth.tar'

        logging(self.checkpoint_latest_path)

    def train_step(self, inputs, targets, step, epoch):

        name, class_ = targets
        loss_train, pred, tp_correct, fp_correct, fn_correct = \
            self.model(inputs, name, batch_label=class_, epoch=epoch)

        loss_train = loss_train / Config.accumulate_step

        loss_train.backward()

        if step % Config.accumulate_step == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss_train.item(), pred, tp_correct, fp_correct, fn_correct

    def train(self):

        best_val_acc = 0
        step_out = 0
        not_best_epoch = 0

        if os.path.exists(self.checkpoint_latest_path):
            print("load latest checkpoint to train!!!!")
            checkpoint = torch.load(self.checkpoint_latest_path)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_schedule'])
            try:
                best_val_acc = checkpoint['best_f1']
                not_best_epoch = checkpoint['not_best_epoch']
            except:
                pass

        self.seqdata = pickle.load(open('{}.pkl'.format(Config.datapre), 'rb'))
        print("load dataset finished!!!!")

        triple_set = Mydataset(self.seqdata['triple_data'], self.seqdata['triple_label'],
                               class_=self.seqdata['triple_class'], is_train=True)

        triple_loader = DataLoader(triple_set, batch_size=Config.batch_size, collate_fn=train_triple_collate_fn,
                                   num_workers=Config.num_workers,
                                   drop_last=True, shuffle=True)

        logging(f'train start:{datetime.datetime.now()}')

        for epoch in range(self.start_epoch + 1, Config.epochs):
            epoch_start = time.time()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            self.model.train()
            train_loss = AverageMeter()
            train_acc = AverageMeter()
            train_recall = AverageMeter()
            train_f1 = AverageMeter()


            for inputs, targets in triple_loader:

                loss, pred, tp_correct_train, fp_correct_train, fn_correct_train = \
                    self.train_step(inputs, targets, step_out, epoch)
                precision_train, recall_train, f1_train = calculate_results(tp_correct_train, fp_correct_train,
                                                                            fn_correct_train)
                train_loss.update(loss, pred.size(0))
                train_acc.update(precision_train, pred.size(0))
                train_recall.update(recall_train, pred.size(0))
                train_f1.update(f1_train, pred.size(0))

                if step_out % (200 * Config.accumulate_step) == 0:
                    log_str = "train epoch: {:3d} step: {:3d} train_loss={:5.2f} train_P={:.5f} train_R={:.5f} train_F1={:.5f}". \
                        format(epoch + 1, step_out, train_loss.avg, train_acc.avg, train_recall.avg, train_f1.avg)
                    logging(log_str)
                step_out += 1

            gc.collect()

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            logging(
                f"Epoch: {epoch} training finished. Time: {epoch_cost}s, speed: {len(triple_loader) / epoch_cost}st/s")

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            precision_val, recall_val, f1_val = self.valids(epoch)
            self.scheduler.step(f1_val)

            is_best = f1_val >= best_val_acc
            best_val_acc = max(f1_val, best_val_acc)
            if is_best:
                not_best_epoch = 0
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_schedule': self.scheduler.state_dict(),
                    'best_f1': best_val_acc,
                    'not_best_epoch': not_best_epoch
                }, self.checkpoint_best_path)

                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_schedule': self.scheduler.state_dict(),
                    'best_f1': best_val_acc,
                    'not_best_epoch': not_best_epoch
                }, self.checkpoint_latest_path)
            else:
                not_best_epoch += 1
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_schedule': self.scheduler.state_dict(),
                    'best_f1': best_val_acc,
                    'not_best_epoch': not_best_epoch
                }, self.checkpoint_latest_path)

            if not_best_epoch > 4:
                break

            if self.optimizer.param_groups[0]['lr'] < Config.min_lr:
                logging("\n!! LR EQUAL TO MIN LR SET.")
                break

        print("Optimization finished!")

    def valids(self, epoch=0):

        val_set = Mydataset(self.seqdata['valid_data'], self.seqdata['valid_label'])

        val_loader = DataLoader(val_set, batch_size=Config.batch_size, collate_fn=collate_fn,
                                num_workers=Config.num_workers,
                                drop_last=True)

        self.model.eval()
        val_loss = AverageMeter()


        true_positive_val, false_positive_val, false_negative_val = 0, 0, 0
        with torch.no_grad():
            for Gs_val, y_val in val_loader:
                loss, pred, tp_correct_val, fp_correct_val, fn_correct_val = self.model(Gs_val, y_val,
                                                                                        teacher_forcing_ratio=0)


                true_positive_val += tp_correct_val
                false_positive_val += fp_correct_val
                false_negative_val += fn_correct_val

                val_loss.update(loss.data, pred.size(0))

        precision_val, recall_val, f1_val = calculate_results(true_positive_val, false_positive_val, false_negative_val)

        log_str = "Val epoch: {:3d} val_loss= {:.5f} val_P={:.5f} val_R={:.5f} val_F1={:.5f}". \
            format(epoch + 1, val_loss.avg, precision_val, recall_val, f1_val)
        logging(log_str)

        return precision_val, recall_val, f1_val

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
    run.train()
    run.test()

