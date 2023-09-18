import numpy as np
import pickle
import math
from os.path import join
from scipy.stats import entropy
import nltk

def dict_update(target,source):
    for key in source:
        if not key in target:
            target[key] = source[key]
        else:
            target_value = target[key]
            source_value = source[key]
            if isinstance(source_value, dict):
                assert isinstance(target_value, dict)
                dict_update(target_value,source_value)
            else:
                assert (type(target_value) == int or float) and (type(source_value) == int or float)
                target[key]= target_value + source_value
    return target
assert str(dict_update({'a':1},{'a':1,'b':20})) == "{'a': 2, 'b': 20}"
assert str(dict_update({'a':1,'c':{'x':100}},{'a':1,'b':20,'c':{'x':300,'y':4000},'z':{'x':50000}})) == "{'a': 2, 'c': {'x': 400, 'y': 4000}, 'b': 20, 'z': {'x': 50000}}"


def listofpairs_compress_with_loss(lst,threshold=0.01):
    maxval = None
    for i in lst:
        if maxval is None or maxval < i[1]:
            maxval = i[1]
    newlist = []
    minval = maxval * threshold
    for i in lst:
        if i[1] >= minval:
            newlist.append(i)
    return newlist
assert str(listofpairs_compress_with_loss([('a',1000),('b',100),('c',10),('d',1)])) == "[('a', 1000), ('b', 100), ('c', 10)]"


def dict_compress_with_loss(dic,threshold=0.01):
    maxval = None
    for d in dic:
        v=dic[d]
        if isinstance(v,dict):
            dict_compress_with_loss(v,threshold) # recursion
        else:
            assert (type(v) == int or float)
            if maxval is None or maxval < v:
                maxval = v
    if maxval is not None:
        todo = []
        minval = maxval * threshold
        for d in dic:
            if dic[d] < minval:
                 todo.append(d)
        for d in todo:
            del dic[d]
    return dic
assert str(dict_compress_with_loss({'a':1000,'b':10,'c':1})) == "{'a': 1000, 'b': 10}"
assert str(dict_compress_with_loss({'x':{'a':1000,'b':10,'c':1},'y':{'m':2000,'n':20,'o':2}})) == "{'x': {'a': 1000, 'b': 10}, 'y': {'m': 2000, 'n': 20}}"


def dict2listsorted(d):
    return [(key, value) for key, value in sorted(d.items())]

def dict_diff(a,b):
    diff = {}
    for key in set(a).union(set(b)):
        if key in a and key in b:
            delta = a[key] - b[key]
            if delta != 0:
                diff[key] = delta
        elif key in a:
            diff[key] = a[key]
        elif key in b: 
            diff[key] = - b[key]
    return diff
assert str(dict2listsorted(dict_diff({'a':1,'b':2,'c':3,'d':1},{'a':1,'b':3,'c':2,'x':1}))) == "[('b', -1), ('c', 1), ('d', 1), ('x', -1)]"

def remove_all(collection,item):
    while item in collection:
        collection.remove(item)

def dictcount(dic,arg,cnt=1):
    if type(arg) == list:

        for i in arg:
            dictcount(dic,i,cnt)
    elif arg in dic:
        dic[arg] = dic[arg] + cnt
    else:
        dic[arg] = cnt

def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def countcount(dic,arg,subarg,cnt=1):
    if arg in dic:
        subdic = dic[arg]
    else:
        dic[arg] = subdic = {}
    dictcount(subdic,subarg,cnt)

def counters_init(max_n):  
    return [{} for n in range(max_n)], [{} for n in range(max_n)], [{} for n in range(max_n)]

def merge_dicts(dicts):
    merged={}
    for d in dicts:
        merged.update(d)
    return merged

def count_subelements(element):
    count = 0
    if isinstance(element, list) or isinstance(element, tuple):
        for child in element:
            count += count_subelements(child)
    elif isinstance(element, dict):
        for key in element:
            count += count_subelements(element[key])
    else:
        count = 1
    return count 
assert count_subelements(['1',2,[[3,'4',{'x':['5',6],'y':(7,'8')},{'z':{'p':9,'q':['10']}}]]]) == 10


# Counting measures 

#https://en.wikipedia.org/wiki/F-score
def dict_precision(ground,guess):
    true_positives = sum([min(guess[key],ground[key]) if key in ground else 0 for key in guess])
    guess_positives = sum(guess.values())
    return true_positives / guess_positives

def dict_recall(ground,guess):
    true_positives = sum([min(guess[key],ground[key]) if key in ground else 0 for key in guess])
    ground_positives = sum(ground.values())
    return true_positives / ground_positives

def list2dict(lst):
    dic = {}
    for l in lst:
        dictcount(dic,l.lower())
    return dic

def listofpairs2dict(lst):
    dic = {}
    for i in lst:
        dictcount(dic,i[0],i[1])
    return dic

def round_str(val,decimals=0):
    if val == 0:
        return '0.'+"".join('0'*decimals)  
    s = str(round(val,decimals))
    point = s.find(".")
    zeros = decimals - (len(s) - point) + 1
    #print(point,len(s),zeros)
    return s + ('0'*zeros)

def calc_f1(ground,guess):
    if isinstance(ground,list):
        ground = list2dict(ground)
    if isinstance(guess,list):
        guess = list2dict(guess)
    true_positives = sum([min(guess[key],ground[key]) if key in ground else 0 for key in guess])
    guess_positives = sum(guess.values())
    ground_positives = sum(ground.values())
    try:
        precision = true_positives / guess_positives
        recall = true_positives / ground_positives
    except:
        print(guess)
        precision = 0
        recall = 0
    return 2 * precision * recall / (precision + recall) if precision > 0 or recall > 0 else 0

def count_word_distance(src_word, src_word_dict, tgt_word_dict):
    total =[]

    for key in tgt_word_dict.keys():
        if key.startswith(src_word) or src_word.startswith(key):
            if abs(len(key) - len(src_word)) ==1 and (len(key) >3 or len(src_word)>3):
               total.append(min(src_word_dict[src_word], tgt_word_dict[key]))
        elif key.endswith(src_word) or src_word.endswith(key):
            if abs(len(key) - len(src_word)) ==1 and (len(key) >3 or len(src_word)>3):
               total.append(min(src_word_dict[src_word], tgt_word_dict[key]))

        else:
            total.append(0)
    return total


def calc_f1_simlarity(ground,guess):
    if isinstance(ground,list):
        ground = list2dict(ground)
    if isinstance(guess,list):
        guess = list2dict(guess)
    tp =[]
    for key in guess:
        if key in ground:
            tp.append(min(guess[key],ground[key]))
        else:
            tp.extend(count_word_distance(key, guess, ground))

    true_positives = sum(tp)
    guess_positives = sum(guess.values())
    ground_positives = sum(ground.values())
    try:
        precision = true_positives / guess_positives
        recall = true_positives / ground_positives
    except:
        print(guess)
        precision = 0
        recall = 0
    return 2 * precision * recall / (precision + recall) if precision > 0 or recall > 0 else 0


def calc_diff(ground,guess):
    if isinstance(ground,list):
        ground = list2dict(ground)
    if isinstance(guess,list):
        guess = list2dict(guess)
    return dict_diff(ground,guess)

def list2matrix(lst):
    rows = 0
    cols = 0
    rows_dict = {}
    cols_dict = {}
    # create labels
    for i in lst:
        row = i[0]
        col = i[1]
        if not row in rows_dict:
            rows_dict[row] = rows
            rows += 1
        if not col in cols_dict:
            cols_dict[col] = cols
            cols += 1
    print(rows,cols)
    print(rows_dict)
    print(cols_dict)
    matrix = np.zeros((rows,cols),dtype=float)
    for i in lst:
        row = i[0]
        col = i[1]
        val = i[2]
        matrix[rows_dict[row],cols_dict[col]] = val
    return sorted(set(rows_dict)), sorted(set(rows_dict)), matrix


def context_save_load(context,context_name,folder='data/temp/'):
    ##https://stackoverflow.com/questions/12544056/how-do-i-get-the-current-ipython-jupyter-notebook-name
    pickle_name = join(folder,context_name)
    if context is None:
        context = pickle.load(open(pickle_name, 'rb'))
    else:
        pickle.dump(context, open(pickle_name, 'wb'), pickle.HIGHEST_PROTOCOL)
    return context


def evaluate_entropy(tokenized_texts):
    """
    Normalized entropy 
    """
    lexicon = {}
    tokens_count = 0
    for tokenized_text in tokenized_texts:
        tokens_count += len(tokenized_text)
        for token in tokenized_text:
            dictcount(lexicon,token)
    distribution = [lexicon[token]/tokens_count for token in lexicon]
    e = entropy(distribution, base=2)
    k = len(lexicon)
    if k > 2:
        e /= math.log2(k)
    return e

def evaluate_anti_entropy(tokenized_texts):
    """
    Normalized anti-entropy 
    """
    return 1.0 - evaluate_entropy(tokenized_texts)


def evaluate_compression(texts,tokenized_texts, nospace=False):
    """
    Coefficient of compression 
    """
    text_len = 0
    tokenized_text_len = 0
    tokens_count = 0
    lexicon = {}
    for text in texts:
        if nospace:
            text_len += len(text.replace(' ', ''))
        else:
            text_len += len(text)
    for tokenized_text in tokenized_texts:
        tokens_count += len(tokenized_text)
        for token in tokenized_text:
            tokenized_text_len += len(token)
            dictcount(lexicon,token)
    tokens_len = 0
    for token in lexicon:
        tokens_len += len(token)
    assert text_len == tokenized_text_len
    return 1.0 - ((tokens_len + tokens_count) / text_len)


