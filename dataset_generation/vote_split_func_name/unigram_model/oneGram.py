import sys
from importlib import reload
from .gram_utils import *

class Dictionary:
    '''
    Dictionary Information and mananger
    '''

    def __init__(self, path):

        # count train.txt words
        self.word_dict = toWordSet(path=path)

        self.N = len(self.word_dict.keys())

    def getPValue(self, word):
        '''
        Calculate the frequency of words
        :param word
        :return: log of the frequency
        '''
        if not self.inDict(word):
            return 1.0 / self.N
        return self.word_dict[word] / self.N

    def inDict(self, word):

        return word in self.word_dict

    def totalNum(self):
        return self.N

    def getDict(self):
        return self.word_dict


class DictionarySmooth:
    '''
    Dictionary Information and mananger with witten-bell smooth
    '''

    def __init__(self, train_path, test_path=None):

        # count train.txt words
        self.word_dict = wittenBellSmoothing(
            unknowWordsSetZero(toWordSet(path=train_path), test_data=test_path))

        self.N = len(self.word_dict.keys())

    def getPValue(self, word):

        return self.word_dict[word]

    def inDict(self, word):
        return word in self.word_dict

    def totalNum(self):
        return self.N

    def getDict(self):
        return self.word_dict


class oneGram(DictionarySmooth):
    '''
    unigram model
    '''

    def __init__(self, train_path, test_path = None, split='back'):

        DictionarySmooth.__init__(self, train_path=train_path, test_path=test_path)

        self.DICT = Dictionary
        self.words = []
        self.value_dict = {}
        self.seg_dict = {}
        self.split_way = split
        self.PUNC =[]

    def backwardSplitSentence(self, sentence, word_max_len=10):

        words = []
        sentence_len = len(sentence)
        range_max = [sentence_len, word_max_len][sentence_len > word_max_len]
        for i in range(range_max - 1):
            words.append((sentence[:i + 1], sentence[i + 1:]))
        return words

    def forwardSplitSentence(self, sentence, word_max_len=10):

        words = []
        sentence_len = len(sentence)
        range_max = [sentence_len, word_max_len][sentence_len > word_max_len]
        for i in range(range_max - 1):
            words.append((sentence[:-(i + 1)], sentence[-(i + 1):]))
        return words

    def maxP(self, sentence):
        '''
        Calculate the maximum probability splitting
        '''

        if len(sentence) <= 1:
            return self.DICT.getPValue(self, sentence)
        # judging the direction of word segmentation：backward or forward
        sentence_split_words = [self.backwardSplitSentence(
            sentence), self.forwardSplitSentence(sentence)][self.split_way != 'back']
        # the maximum probability value
        max_p_value = 0
        word_pairs = []
        word_p = 0

        for pair in sentence_split_words:
            p1, p2 = 0, 0

            if pair[0] in self.value_dict:
                p1 = self.value_dict[pair[0]]
            else:
                p1 = self.maxP(pair[0])

            if pair[1] in self.value_dict:
                p2 = self.value_dict[pair[1]]
            else:
                p2 = self.maxP(pair[1])

            word_p = p1 * p2

            if max_p_value < word_p:
                max_p_value = word_p
                word_pairs = pair
        # Query the frequency corresponding to the current sentence in the dictionary, if it does not exist, return 1/N
        sentence_p_value = self.DICT.getPValue(self, sentence)

        # When the probability of not splitting is maximum, update each value
        if sentence_p_value > max_p_value and self.DICT.inDict(self, sentence):
            self.value_dict[sentence] = sentence_p_value
            self.seg_dict[sentence] = sentence
            return sentence_p_value
        # When the probability of a combination of segmentation is the highest,
        # update the corresponding probability of the sentence
        # to avoid repeated calculations for subsequent segmentation
        else:
            self.value_dict[sentence] = max_p_value
            self.seg_dict[sentence] = word_pairs
            return max_p_value

    def getSeg(self):
        return self.seg_dict

    def segment(self, sentence):

        words = []
        sentences = []
        # If it contains punctuation, it is separated by punctuation
        start = -1
        for i in range(len(sentence)):
            if sentence[i] in self.PUNC and i < len(sentence):
                sentences.append(sentence[start + 1:i])
                sentences.append(sentence[i])
                start = i
        if not sentences:
            sentences.append(sentence)

        for sent in sentences:
            self.maxP(sent)
            words.extend(self.dumpSeg(sent))

        words_index =[]
        cur_index = 0

        for word in words:
            start_idx = sentences[0].find(word, cur_index)

            end_idx = start_idx+ len(word)
            cur_index = end_idx
            words_index.append((start_idx, end_idx-1))

        return words_index

    def dumpSeg(self, sentence):

        words = []
        if sentence in self.seg_dict:
            pair = self.seg_dict[sentence]
            if isinstance(pair, tuple):
                words.extend(self.dumpSeg(pair[0]))
                words.extend(self.dumpSeg(pair[1]))
            else:
                if pair == sentence:
                    words.append(pair)
                else:
                    words.extend(self.dumpSeg(pair))
        else:
            words.append(sentence)
        return words
