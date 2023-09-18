

def prefixed_match_from_list(lst, text):
    item_dict = {}
    for item in lst:
        if text.startswith(item[0]):
            return item

    return None


def prefixed_match(prefixed_dict, text):
    '''
    if len(text) > 1:
        letter = text[0]
    else:
        letter = text
    '''
    letter = text[0]
    if not letter in prefixed_dict:
        return None
    return prefixed_match_from_list(prefixed_dict[letter], text)


def tokenize_with_prexied_sorted_lexicon(prefixed_dict, text, cased=False):
    original = text
    if cased:  # if need to spend time on lowercasing non-lowercased text
        text = text.lower()
    tokens = []
    start = 0
    cur = 0
    length = len(text)
    sum_weight = 0
    while cur < length:
        subtext = text[cur:]
        word_weight = prefixed_match(prefixed_dict, subtext)
        # print(al)
        if not word_weight is None:
            word_len = len(word_weight[0])
            if start < cur:
                tokens.append(original[start:cur])
            tokens.append(original[cur:cur + word_len])
            sum_weight += word_weight[1]
            cur += word_len
            start = cur
        else:
            cur += 1
            # print('yo')
    if start < cur:
        tokens.append(original[start:cur])
        # print(original[start:cur])
    return tokens, sum_weight
