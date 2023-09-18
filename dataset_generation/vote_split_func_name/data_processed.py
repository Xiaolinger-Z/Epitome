from os import listdir
from os.path import isfile, join
import datetime as dt
from re import compile, VERBOSE
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

RE_WORDS = compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+  # Numbers
    ''', VERBOSE)

def get_split_subtokens_proc_name(s):
    # get sub-tokens. Remove them if len()<1 (single letter or digit)
    return [x for x in [str(x).lower() for x in RE_WORDS.findall(s)] if not x.isdigit()]

def tokenizer_train_folder(self,path,debug=True):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    t1 = dt.datetime.now()
    #i = 1
    for file in onlyfiles:
        t2 = dt.datetime.now()
        with open(join(path, file)) as f:
            lines = f.readlines()
            self.train(lines)
        t3 = dt.datetime.now()
        if debug:
            print("{}\t{}\t{}".format(join(path, file),len(lines),(t3-t2).seconds))
        #i += 1
        #if i > 2:
        #    break
    t4 = dt.datetime.now()
    if debug:
        print("{}\t{}:{}".format(self.count_params(),round((t4-t1).seconds / 60),((t4-t1).seconds % 60)))

def sort_dict(a_dict, option="value"):
    '''
    Sort the dict
    '''
    if option in ["value", "key"]:
        result_dict = {}
        if option == "key":
            temp_list = list(a_dict.keys())
            temp_list.sort()
            for item in temp_list:
                result_dict[item] = a_dict[item]
        else:
            temp_value_list = list(a_dict.values())
            temp_key_list = list(a_dict.keys())
            for i in range(len(temp_key_list)):
                for j in range(len(temp_key_list) - i - 1):
                    if temp_value_list[j] > temp_value_list[j + 1]:
                        temp = temp_key_list[j]
                        temp_key_list[j] = temp_key_list[j + 1]
                        temp_key_list[j + 1] = temp
                        temp = temp_value_list[j]
                        temp_value_list[j] = temp_value_list[j + 1]
                        temp_value_list[j + 1] = temp
            for key, value in zip(temp_key_list, temp_value_list):
                result_dict[key] = value
        return result_dict
    raise ValueError(option + " is not in option list——[key,value]")


def get_pos(treebank_tag):
    """
    get the pos of a treebank tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement

def lemmatize_name(words):
    '''
    words = []
    for word in res:
        if not isinstance(word, str) or word == '':
            continue
        words.append(word)
    '''
    lem = WordNetLemmatizer()
    tokens = nltk.pos_tag(words)
    res = []
    for word, tag in tokens:
        if word in nltk.corpus.stopwords.words('english'):
            res.append(word)
        else:
            wntag = get_pos(tag)
            if wntag is None:  # not supply tag in case of None
                word = lem.lemmatize(word)
            else:
                word = lem.lemmatize(word, pos=wntag)
            res.append(word)

    return res

class Remove_ida_pattern:

    def __init__(self):

        # regexes
        self.underprefix = re.compile(r'^_+')
        self.undersuffix = re.compile(r'_+$')
        self.bitssuffix = re.compile(r'(32|64)$')
        self.bitsprefix = re.compile(r'^(32|64)')

        self.r2_prefix = re.compile(r'^sym\.')
        self.r2_dyn_prefix = re.compile(r'^sym\.imp\.')

        self.isra = re.compile(r'\.isra(\.\d*)*')
        self.part = re.compile(r'\.part(\.\d*)*')
        self.constprop = re.compile(r'\.constprop(\.\d*)*')
        self.constp = re.compile(r'\.constp(\.\d*)*')

        # self.libc           = re.compile(r'libc\d*')
        self.sse2 = re.compile(r'_sse\d*')
        self.ssse3 = re.compile(r'_ssse\d*')
        self.avx = re.compile(r'avx\d*')
        self.cold = re.compile(r'\.cold$')

        self.unaligned = re.compile(r'unaligned')
        # self.internal = re.compile(r'internal')
        self.erms = re.compile(r'erms')
        self.__ID__ = re.compile(r'_+[A-Z]{1,2}_+')

        self.dot_prefix = re.compile(r'^\.+')
        self.dot_suffix = re.compile(r'\.+$')
        self.num_suffix = re.compile(r'_+\d+$')
        self.num_prefix = re.compile(r'^\d+_+')
        self.dot_num_suffix = re.compile(r'\.+\d+$')
        self.num_only_prefix = re.compile(r'^\d+')
        self.num_only_suffix = re.compile(r'\d+$')

        self.repeated_nonalpha = re.compile(r'([^a-zA-Z0-9\d])\1+')

        self.ida_import = re.compile(r'__imp_')
        self.data_lib = re.compile(r'@@.*')


    def strip_library_decorations(self, name):
        """
            Compare names of symbols against known prefixed and suffixes
            strcpy -> __strcpy
            open -> open64
        """
        content_replace = [
            self.__ID__, self.ssse3, self.sse2,  self.avx, self.cold, self.unaligned, self.erms,
            self.constprop, self.constp, self.isra, self.part
        ]

        syntax_replace = [
                self.r2_dyn_prefix, self.r2_prefix,
            self.dot_num_suffix, self.num_suffix, self.num_prefix,
            #self.bitssuffix, self.bitsprefix,
            self.dot_prefix, self.dot_suffix,
            self.underprefix, self.undersuffix,
            self.num_only_prefix #, self.num_only_suffix
        ]

        for cf in content_replace:
            name = re.sub(cf, "", name)
            for sf in syntax_replace:
                name = re.sub(sf, "", name)

            name = re.sub(self.repeated_nonalpha, '\g<1>', name)

        name = get_split_subtokens_proc_name(name)
        name = lemmatize_name(name)

        return ' '.join(name)