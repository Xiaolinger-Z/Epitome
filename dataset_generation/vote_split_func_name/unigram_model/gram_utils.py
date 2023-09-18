from collections import defaultdict
from os import listdir
from os.path import isfile, join
import datetime as dt
import joblib


def toWordSet(path, is_save=False, save_file='wordSet.pkl'):
    '''
    get word dictionary
    '''
    word_dict = defaultdict(float)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # i = 1
    for file in onlyfiles:
        t2 = dt.datetime.now()
        with open(join(path, file)) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().replace("_", " ")
                word_list_lower = line.lower().split()
                for word in word_list_lower:
                    if word.isdigit():
                        continue
                    if len(word) < 2:
                        continue
                    word_dict[word]+=1

    if is_save:
        joblib.dump(word_dict, save_file)

    print("successfully get word dictionary!")
    print("the total number of words is:{0}".format(len(word_dict.keys())))
    return word_dict

def toWordSet_test(file_name, is_save=False, save_file='wordSet.pkl'):
    '''
     get word dictionary
    '''

    word_dict = defaultdict(float)

    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            _, line = line.strip().split('\t')
            line = line.strip().replace("_", " ")
            word_list_lower = line.lower().split()
            for word in word_list_lower:
                if word.isdigit():
                    continue
                if len(word) < 2:
                    continue
                word_dict[word] += 1
    if is_save:
        joblib.dump(word_dict, save_file)

    print("successfully get word dictionary!")
    print("the total number of words is:{0}".format(len(word_dict.keys())))
    return word_dict

def wittenBellSmoothing(word_dict):
    '''
    witten-Bell smoothing
    Pi =
         T/{(N+T)Z}  if Ci == 0
         Ci/(N+T)    if Ci != 0
    :param word_dict:
    :return:
    '''
    Z, N, T = 0, 0, 0
    for key in word_dict:
        if word_dict[key] == 0.:
            Z += 1
        else:
            N += word_dict[key]
            T += 1
    # 平滑处理
    re_word_dict = word_dict.copy()
    for key in word_dict:
        if word_dict[key] == 0.:
            re_word_dict[key] = T / ((N + T) * Z)
        else:
            re_word_dict[key] = word_dict[key] / (N + T)
    print('successfully witten-Bell smoothing!')

    return re_word_dict

def unknowWordsSetZero(word_dict, test_data=None):
    '''
    count oov words, add in word dict，value = 0
    :param word_dict:
    :param file_name:
    :return:
    '''
    if test_data is not None:
        test_word_dict = toWordSet_test(test_data)
        unknow_word_cnt = 0
        for test_word in test_word_dict:
            if test_word not in word_dict:
                word_dict[test_word] = 0.
                unknow_word_cnt += 1

        print('unknow words count number: {0}, set unknow word value = {1}'.format(
            unknow_word_cnt, 0.0))

    return word_dict