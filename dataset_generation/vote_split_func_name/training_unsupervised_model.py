
from os import listdir
from os.path import isfile, join
import argparse
from unsupervised_learning.util import *
from unsupervised_learning.token import *
from data_processed import tokenizer_train_folder,  sort_dict, get_split_subtokens_proc_name



def train(datapath, save_path):
    model = FreedomTokenizer(max_n=10, mode='chars', debug=False)

    tokenizer_train_folder(model, datapath)
    model.store(save_path)

def train_lex(datapath, save_path):

    onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]

    word_count =  dict()
    for file in onlyfiles:

        with open(join(datapath, file)) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace("_", " ")
                word_list_lower = line.lower().split()
                for word in word_list_lower:
                    if word.isdigit():
                        continue
                    if len(word)<2:
                        print(word)
                        continue
                    if word in word_count.keys():
                        word_count[word]+=1
                    else:
                        word_count[word] = 1

    word_count = sort_dict(word_count, option="key")
    with open(save_path, "w") as fo:
        for key, value in word_count.items():
            fo.write(key +'\t' + str(value) +'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train unsupervised model')
    parser.add_argument('--dataset_path', type=str, nargs=1,
                        help='directory where dataset for unsupervised model training')
    parser.add_argument('--save_path', type=str, nargs=1,
                        help='path for freedom model')
    parser.add_argument('--lexicon_path', type=str, nargs=1,
                        help='path for saving lexicon dictionary',)

    args = parser.parse_args()
    output_dir = args.output_dir[0]
    train(args.dataset_path, args.save_path)
    train_lex(args.dataset_path, args.lexicon_path)
