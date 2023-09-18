from torch.utils.data import Dataset
import tqdm
import torch
import random
import pickle as pkl
import numpy as np


class BARTDataset(Dataset):
    def __init__(self, cfg_corpus_path, dfg_corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        #self.cfg_corpus_lines = corpus_lines
        self.corpus_lines = corpus_lines
        self.dfg_corpus_path = dfg_corpus_path
        self.cfg_corpus_path = cfg_corpus_path
        self.encoding = encoding
        self.dfg_lines=[]
        self.cfg_lines=[]
        dfg_corpus_lines = 0
        cfg_corpus_lines = 0
        # load DFG sequences

        with open(dfg_corpus_path, "r", encoding=encoding) as f:
                if self.corpus_lines is None and not on_memory:
                    for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        # self.corpus_lines += 1
                        dfg_corpus_lines += 1

                if on_memory:
                    dfg_lines = [line.strip().split("\t")
                                  for line in tqdm.tqdm(f, desc="Loading DFG Dataset", total=corpus_lines)]
                    self.dfg_lines.extend(dfg_lines)
        if on_memory:
            self.corpus_lines = len(self.dfg_lines)

       # load CFG sequences
        with open(cfg_corpus_path, "r", encoding=encoding) as f:
                if self.corpus_lines is None and not on_memory:
                    for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                        # self.corpus_lines += 1
                        cfg_corpus_lines+=1
                if on_memory:
                    cfg_lines = [line.strip().split("\t")
                                  for line in tqdm.tqdm(f, desc="Loading CFG Dataset", total=corpus_lines)]
                    self.cfg_lines.extend(cfg_lines)

        if on_memory:
            if self.corpus_lines > len(self.cfg_lines):
                self.corpus_lines = len(self.cfg_lines)
        else:
            if dfg_corpus_lines > cfg_corpus_lines:
                self.corpus_lines = cfg_corpus_lines
            else:
                self.corpus_lines = dfg_corpus_lines

        if not on_memory:
            self.cfg_file = open(cfg_corpus_path, "r", encoding=encoding)
            self.dfg_file = open(dfg_corpus_path, "r", encoding=encoding)
            self.cfg_random_file = open(cfg_corpus_path, "r", encoding=encoding)
            self.dfg_random_file = open(dfg_corpus_path, "r", encoding=encoding)

            for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.cfg_random_file.__next__()
            for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.dfg_random_file.__next__()



    def __len__(self):
        return self.corpus_lines


    def __getitem__(self, item):
        

        c1, c2, c_label, d1, d2, d_label = self.random_bert_sent(item)

        cfg_output = self.vocab(c1,c2, add_special_tokens=True, return_attention_mask=False, truncation=True, max_length=self.seq_len, return_token_type_ids=True)
        dfg_output = self.vocab(d1,d2, add_special_tokens=True, return_attention_mask=False, truncation=True, max_length=self.seq_len, return_token_type_ids=True)


        output = {"dfg_bert_input": dfg_output,
                  "dfg_is_next": d_label,
                  "cfg_bert_input": cfg_output,
                  "cfg_is_next": c_label
                  }


        return output

    def random_bert_sent(self, index):
        c1, c2, d1, d2 = self.get_bert_corpus_line(index)
        dice = random.random()  # TODO: should throw the dice twice here.
        if dice > 0.25:
            return c1, c2, 1, d1, d2, 1
        elif 0.25 <= dice < 0.5:
            return c1, self.get_random_line(), 0, d1, d2, 1
        elif 0.5 <= dice < 0.75:
            return c1, c2, 1, d2, d1, 0
        else:
            return c1, self.get_random_line(), 0, d2, d1, 0

    def get_bert_corpus_line(self, item):
        if self.on_memory:
            return self.cfg_lines[item][0], self.cfg_lines[item][1], self.dfg_lines[item][0], self.dfg_lines[item][1]

        # now only on_memory copurs are supported
        else:
            try:
                cfg_line = self.cfg_file.__next__()
            except StopIteration:

                # if cfg_line is None:
                self.cfg_file.close()
                self.cfg_file = open(self.cfg_corpus_path, "r", encoding=self.encoding)
                cfg_line = self.cfg_file.__next__()
            try:
                dfg_line = self.dfg_file.__next__()
            except StopIteration:
                # if dfg_line is None:
                self.dfg_file.close()
                self.dfg_file = open(self.dfg_corpus_path, "r", encoding=self.encoding)
                dfg_line = self.dfg_file.__next__()

            # print(r"%s"%(cfg_line.strip()))

            t1, t2 = cfg_line.strip().split("\t")
            t3, t4 = dfg_line.strip().split("\t")
            return t1, t2, t3, t4

    def get_random_line(self):
        if self.on_memory:
            l = self.cfg_lines[random.randrange(len(self.cfg_lines))]
            return l[1]

        # now only on_memory copurs are supported
        try:
            cfg_line = self.cfg_file.__next__()

        except StopIteration:
             self.cfg_file.close()
             self.cfg_file = open(self.dfg_corpus_path, "r", encoding=self.encoding)
             for _ in range(np.random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                 self.cfg_random_file.__next__()
             cfg_line = self.cfg_random_file.__next__()

        return cfg_line[:-1].split("\t")[1]
