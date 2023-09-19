import numpy as np
import itertools
from gensim.models import Word2Vec, FastText
import pandas as pd
from threading import Lock
import enchant
import nltk
from nltk.corpus import wordnet as wn


class SmithWaterman:
    """
        Implements Smith-Waterman distance
    """
    def __init__(self, gap_cost=2, match_score=3):
        self.match_score    = match_score
        self.gap_cost       = gap_cost

    def matrix(self, a, b):
        """
            Calculates similarity matrix for 2 sequences
        """
        H = np.zeros((len(a)+1, len(b)+1), np.int)

        for i, j in itertools.product(range(1, H.shape[0]), range(1, H.shape[1])):
            match   = H[i-1, j-1] + self.match_score if a[i-1] == b[j-1] else - self.match_score
            delete  = H[i-1, j] - self.gap_cost
            insert  = H[i, j-1] - self.gap_cost

            H[i, j] = max(match, delete, insert, 0)

        return H

    def traceback(self, H, b, b_='', old_i=0):
        """
            Recursivly find best alignment
        """
        H_flip = np.flip(np.flip(H, 0), 1)
        i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
        i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))  # (i, j) are **last** indexes of H.max()

        if H[i, j] == 0:
            return b_, j

        b_ = b[j - 1] + '-' + b_ if old_i - i > 1 else b[j - 1] + b_
        return self.traceback(H[0:i, 0:j], b, b_, i)

    def max_alignment(self, a, b):
        """
            Returns the string indicies for the highest scoring alignment
            :return: start_ind, end_ind
            :rtype: tuple of 2 ints
        """
        a, b    = a.lower(), b.lower()
        H       = self.matrix(a, b)
        b_, pos = self.traceback(H, b)
        return pos, pos + len(b_)

    def score(self, a, b):
        """return a similarity score between 2 sequences"""
        start, end = self.max_alignment(a, b)
        assert(end >= start)
        return end-start

    def edit_similarity(self, a, b):

        levehstien_distance = self.score(a, b)

        m = float(max(len(a), len(b)))
        edit_sim = float(levehstien_distance) / m
        return edit_sim

    def distance(self, a, b):
        """
            Get the distance between a and b, smaller is closer distance
        """
        d = self.score(a, b)
        return 1.0 / (1.0 + np.log(1+d))

class NLPSimilarity:

    def __init__(self):

        self.enchant_lock = Lock()
        self.us_D = enchant.Dict("en_US")
        self.gb_D = enchant.Dict("en_GB")
        self.WORD_MATCH_THRESHOLD = 0.36787968862663154641

    def compare_synsets(self, A, B):
        """
            Compare two sets of words based on synsets from wordnet
        """
        _A = list(A)
        _B = list(B)

        mc = 0.0
        ##score is a function of teh maximum length, if length of 2 words sets is 10, 2 and all 2 are in 10 -> wrong name
        max_m = float( max(len(_A), len(_B)) )
        if max_m == 0.0:
            raise Exception("Error, empty word set passed ({},{})".format(A, B))

        a_synsets = list(map(lambda x: wn.synsets(x), _A))
        b_synsets = list(map(lambda x: wn.synsets(x), _B))

        for i, a_ss in enumerate(a_synsets):
            WORD_MATCH = False
            a_words_list = list(map(lambda x: x.lemma_names(), a_ss))
            a_words = [word for word_list in a_words_list for word in word_list]
            a_synonyms = set(a_words)
            for j, b_ss in enumerate(b_synsets):
                b_words_list = list(map(lambda x: x.lemma_names(), b_ss))
                b_words = [word for word_list in b_words_list for word in word_list]
                b_synonyms = set(b_words)
                if len(a_synonyms.intersection(b_synonyms)) > 0:
                    #print("Matched ({},{}) on {}".format(_A[i], _B[j], a_synonyms.intersection(b_synonyms)))
                    WORD_MATCH = True
                    break

            if WORD_MATCH:
                mc += 1.0

        return mc / max_m

    def check_word_similarity(self, correct_word, inferred_word):

        if correct_word.startswith(inferred_word) or inferred_word.startswith(correct_word):
            # print("Matched on substring! {} -> {}".format(inferred_name, correct_name))
            return True

        smith = SmithWaterman()
        edit_sim = smith.edit_similarity(correct_word, inferred_word)
        if edit_sim > float(2 / 3):
            return True

        self.enchant_lock.acquire()

        try:
            if self.us_D.check(correct_word) or self.gb_D.check(correct_word):
                correct_word = set(correct_word)
            else:
                return False

            if self.us_D.check(inferred_word) or self.gb_D.check(inferred_word):
                inferred_word = set(inferred_word)
            else:
                return False

        except Exception as e:
            print("[!] Could not compute check_similarity_of_symbol_name( {} , {} )".format(correct_word, inferred_word))

        finally:
            self.enchant_lock.release()

        stemmer = nltk.stem.PorterStemmer()
        lemmatiser = nltk.stem.wordnet.WordNetLemmatizer()

        stemmed_inferred = set(map(lambda x: stemmer.stem(x), correct_word))
        stemmed_correct = set(map(lambda x: stemmer.stem(x), inferred_word))

        lemmatised_inferred = set(map(lambda x: lemmatiser.lemmatize(x), correct_word))
        lemmatised_correct = set(map(lambda x: lemmatiser.lemmatize(x), inferred_word))

        if len(lemmatised_correct) > 0 and len(lemmatised_inferred) > 0:
            jaccard_distance = nltk.jaccard_distance(lemmatised_correct, lemmatised_inferred)
            if jaccard_distance < self.WORD_MATCH_THRESHOLD:
                # print("\tJaccard Distance Lemmatised {} : {} -> {}".format(jaccard_distance, inferred_name, correct_name))
                return True

        if len(stemmed_correct) > 0 and len(stemmed_inferred) > 0:
            jaccard_distance = nltk.jaccard_distance(stemmed_correct, stemmed_inferred)
            if jaccard_distance < 1.0 - self.WORD_MATCH_THRESHOLD:
                # print("\tJaccard Distance Stemmed {} : {} -> {}".format(jaccard_distance, inferred_name, correct_name))
                return True

        if len(correct_word) > 0 and len(inferred_word) > 0:
            if self.compare_synsets(correct_word, inferred_word) >= 0.385:
                # print("\tMatched on wordnet synsets: {} -> {}".format(inferred_name, correct_name))
                return True

        return False


if __name__=="__main__":
    '''
    smith = SmithWaterman()
    correct_name = 'subdomain'
    predict_name = 'sub-domains'
    levehstien_distance = smith.score(correct_name, predict_name)

    m = float(max(len(correct_name), len(predict_name)))
    edit_sim = float(levehstien_distance) / m
    print(edit_sim)
    '''
    model = Word2Vec.load("./models/cbow_50.model")
    Words_Vectors = pd.DataFrame(model.wv.vectors)
    print(Words_Vectors.shape)
    Words_Vectors.head()