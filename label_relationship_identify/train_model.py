from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.utils import tokenize
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import argparse
import pandas as pd
from compute_similarity import SmithWaterman, NLPSimilarity
import nltk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-path", help="file path to the training data", type=str, default='')
    parser.add_argument("--model-name",
                        help="name of the model to be trained, options: ['skip_gram', 'fasttext']", type=str,
                        default='fasttext')
    parser.add_argument("--num-epochs", help="number of epochs to train the model", type=int, default=50)
    args = parser.parse_args()
    training_set = args.training_data_path
    train_one_model(args.model_name, training_set, args.num_epochs)

    model_path1 = f"models/skip_gram_{args.num_epochs}.model"
    model_path2 = f"models/fasttext_{args.num_epochs}.model"

    word_dict =dict()
    for model_path in [model_path1, model_path2]:
        tmp_dict = get_similarity_word_dict(model_path)
        for key, vaule in tmp_dict.items():
            if key in word_dict:
                word_dict[key].extend(vaule)
            else:
                word_dict[key] = vaule

    final_word_dict = vote_similarity_word(word_dict)

    out_path = "./modelout/related_word_{}.txt".format(args.arch)
    with open(out_path, 'w') as fo:
        for key, vaules in final_word_dict.items():
            out_str = ''
            if len(vaules) < 1:
                continue
            for vaule in vaules:
                out_str += vaule + ' '
            fo.write(key + '\t' + out_str + '\n')


def load_model(model_name, num_epochs):
    model_path = f"models/{model_name}_{num_epochs}.model"
    if model_name == "fasttext":
        model = FastText.load(model_path)
    else:
        model = Word2Vec.load(model_path)
    return model


class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        print(f"epoch: {self.epoch}")
        self.epoch += 1


def train_one_model(model_name, training_set, num_epochs):
    def create_model(model_name):
        if model_name == "skip_gram":
            model = Word2Vec(size=100, window=5, min_count=1, workers=30, negative=5, sg=1)
        elif model_name == "fasttext":
            model = FastText(size=100, window=5, min_count=1, workers=30, negative=5, sg=1)
        else:
            model = None
        return model

    corpus_file = datapath(training_set)

    print(f"Train {model_name} with {num_epochs} epochs")
    print(f"Create {model_name}")
    model = create_model(model_name)
    model.build_vocab(corpus_file=corpus_file)
    total_words = model.corpus_total_words
    print("# total words:", total_words)

    model.train(corpus_file=corpus_file, total_words=total_words, epochs=num_epochs, report_delay=1,
                compute_loss=True,  # set compute_loss = True
                callbacks=[callback()])
    model_path = f"models/{model_name}_{num_epochs}.model"

    model.save(model_path)
    print(f"Save {model_name} to {model_path}")

def get_similarity_word_dict(model_path):
    word_dict = dict()
    if 'fasttext' in model_path:
        model = FastText.load(model_path)
    else:
        model= Word2Vec.load(model_path)

    words = pd.DataFrame.from_dict(model.wv.vocab, orient='index', columns=['words']).index
    #words = ['str', 'func', 'exec', 'buf', 'addr', 'address', 'setup', 'initialize']
    for word in words:
        sims = model.wv.most_similar(word, topn=5)
        tmp =[]
        for i, _ in sims:
            tmp.append(i)

        word_dict[word] = tmp

    return word_dict

def count_repet(words):
    myset = set(words)
    out =[]
    for item in myset:
        num = words.count(item)
        if num > 1:
            out.append(item)
    for i in out:
        myset.remove(i)

    return out, list(myset)

def determine_word(key, value):
    out =[]
    count_simlilarity=NLPSimilarity()
    for i in value:
        if count_simlilarity.check_word_similarity(key, i):
            out.append(i)

    return out


def vote_similarity_word(words_dict):
    final_words_dict = {}

    for key, value in words_dict.items():
        tmp_value =[]
        out, pend_value = count_repet(value)
        tmp_value.extend(out)
        final_value = determine_word(key, pend_value)
        tmp_value.extend(final_value)
        final_words_dict[key] = tmp_value

    return final_words_dict


if __name__ == "__main__":
    main()

