import os, sys
import pandas as pd
from config import Config
import glob
from intervaltree import Interval, IntervalTree

from vote_split_func_name.unsupervised_learning.token import FreedomTokenizer, LexiconIndexedTokenizer, weightedlist2dict, \
    evaluate_tokenizer_merge_f1
from vote_split_func_name.unsupervised_learning.util import listofpairs_compress_with_loss, remove_all
from vote_split_func_name.unsupervised_learning.text import delimiters
from vote_split_func_name.unigram_model.oneGram import oneGram
import vote_split_func_name.rules_split.sorted_NLP as sorted_NLP
from vote_split_func_name.data_processed import Remove_ida_pattern, lemmatize_name

# the rule-based meyhod
nlp = sorted_NLP.NLP(Config)
# remove the string added by analyse tool
remove_pattern = Remove_ida_pattern()

# the unigram model
OG = oneGram(train_path=Config.unigram_train_path)

# unsupervised learning method
Freedombase = FreedomTokenizer(name=Config.Freedom_model,
                               max_n=10, mode='chars', debug=False)

# lexicon tokenizer, part of unsupervised learning method
en_lex = list(pd.read_csv(Config.Freedom_lexicon_model, sep='\t', header=None, na_filter=False).to_records(index=False))

# merge and get top weight
en_lex_dict = weightedlist2dict(en_lex, lower=True)  # with case-insensitive merge
top_weight = max([en_lex_dict[key] for key in en_lex_dict], key=lambda item: item)
# add delimiters to the list
en_lex_delimited = en_lex + [(i, top_weight) for i in list(delimiters)]

t = 0.00001
lex = listofpairs_compress_with_loss(en_lex_delimited, t) if t > 0 else en_lex
en_lex_tokenizer = LexiconIndexedTokenizer(lexicon=lex, sortmode=2, cased=True)  # the performan is best


def _score_abbrs(final_pred, out_pred):
    t = IntervalTree()
    used = set()  # set([])

    for start, end in final_pred:

        if t.overlaps(start, end):
            continue
        if start != end:
            t.addi(start, end, int(end - start))

            used.add((start, end))

    for start, end in out_pred:

        if t.overlaps(start, end):
            continue
        if start != end:
            t.addi(start, end, int(end - start))

            used.add((start, end))

    return used


def best_split_name(g_pred, r_pred, p_pred, final_pred):
    place_1 = set(g_pred).intersection(set(final_pred))
    place_2 = set(r_pred).intersection(set(final_pred))
    place_3 = set(p_pred).intersection(set(final_pred))
    if len(place_1) > len(place_2):
        if len(place_3) > len(place_1):
            out = p_pred
        else:
            out = g_pred
    else:
        if len(place_3) > len(place_2):
            out = p_pred
        else:
            out = r_pred

    return _score_abbrs(final_pred, out)  # set(out).union(final_pred)


def decide_split_place(g_pred, r_pred, p_pred, name):
    place_1 = set(g_pred).intersection(set(r_pred))
    place_2 = set(g_pred).intersection(set(p_pred))
    place_3 = set(r_pred).intersection(set(p_pred))
    tmp_place = place_1.union(place_2)
    tmp_place = tmp_place.union(place_3)
    final_place = best_split_name(g_pred, r_pred, p_pred, tmp_place)

    final_place = sorted(list(final_place))

    out = []
   
    pre_end_idx = 0
    for idx, value in enumerate(final_place):
        star_idx, end_idx = value
        if idx > 0:
            if pre_end_idx + 1 != star_idx:
                out.append(name[pre_end_idx + 1:star_idx])
            pre_end_idx = end_idx
        else:
            if star_idx != 0:
                out.append(name[0:star_idx])
            pre_end_idx = end_idx

        out.append(name[star_idx:end_idx + 1])

    if pre_end_idx != len(name) - 1:
        out.append(name[pre_end_idx + 1:len(name)])

    return out


def filter_only_digit(words):
    remove_all(words, ' ')

    out = []

    for word in words:
        if not word.isdigit():
            if len(word.strip()) > 0:
                out.append(word.strip())

    return out


out_func_name = []
# compiler commom function

for program in Config.PORGRAM_ARR:
    for mode in Config.MODE:
        for data_name in Config.NAME:
            temp_idb = Config.FEA_DIR + os.sep + program + os.sep + mode + os.sep + data_name
            
            for version in os.listdir(temp_idb):
                curBinDir = temp_idb + os.sep + str(version)

                filters = glob.glob(curBinDir + os.sep + "raw_*_graph_labels.txt")

                for file_name in filters:  # os.listdir(temp_idb):

                    if 'graph_labels' in str(file_name):
                        base_name = os.path.basename(file_name)

                        base_name = base_name.split('_')[1:-3]

                        name_output = []

                        if not os.path.exists(file_name):
                            continue

                        with open(file_name, 'r') as f:
                            text = f.readlines()

                            for readline in text:
                                demangle_name = readline.strip()

                                name = remove_pattern.strip_library_decorations(demangle_name)
                                g_pred = OG.segment(name)
                                r_pred = nlp.canonical_name_index(name)
                                p_pred = evaluate_tokenizer_merge_f1(Freedombase.model, en_lex_tokenizer, name, 'peak-',
                                                                     'peak+', [1, 2, 3, 4, 5], 0.1,
                                                                     spaces=False)
                                out_pred = decide_split_place(g_pred, r_pred, p_pred, name)
                                out_pred = filter_only_digit(out_pred)

                                out = []
                                for abbr in out_pred:
                                    out.append(nlp.expand_abbreviations(abbr))

                                final_out = lemmatize_name(out)
                                name_output.append('_'.join(final_out))

                                print(" {} ------> {}".format(demangle_name, '_'.join(final_out)))

                            # print(mod_res)

                        rewrite_path = file_name.replace('raw_', '', 1)
                        # rewrite_path = ""
                        with open(rewrite_path, 'w') as fo:
                            # fo.write('')
                            for name in name_output:
                                fo.write(name.replace(' ', '_') + '\n')

