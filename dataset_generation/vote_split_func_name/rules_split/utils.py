import copy

def sorted_label_token(abbr_dict):
    keys = abbr_dict.keys()
    keys =sorted(keys)

    return [abbr_dict[key] for key in keys]


def find_abbr_index(all_index_dict, max_len):
    fianl_index = []
    tmp_index_dict = copy.deepcopy(all_index_dict)
    for key, value in all_index_dict.items():
        if len(value) > 1:

            tmp_value = copy.deepcopy(value)
            for tmp_i, tmp_j in value:
                for f_value in all_index_dict.values():
                    for strat_i, end_i in f_value:
                        if strat_i <= tmp_i and tmp_j < end_i:
                            try:
                                tmp_value.remove((tmp_i,tmp_j))
                            except:
                                pass
                        elif strat_i < tmp_i and tmp_j <= end_i:
                            try:
                                tmp_value.remove((tmp_i,tmp_j))
                            except:
                                pass
                        elif strat_i < tmp_i and tmp_j < end_i:
                            try:
                                tmp_value.remove((tmp_i,tmp_j))
                            except:
                                pass
        else:
            tmp_value = value

        tmp_index_dict[key] = tmp_value

    pre_end_idx =0
    for value in tmp_index_dict.values():
        if len(value)>0:
            star_idx, end_idx = value[0]
            fianl_index.append((star_idx, end_idx-1))

    fianl_index = sorted(fianl_index,key= lambda tup:tup[0])

    output_index =[]
    for idx, value in enumerate(fianl_index):
        star_idx, end_idx = value
        if idx > 0:
            if pre_end_idx+1 < star_idx:
                output_index.append((pre_end_idx+1, star_idx-1))

        else:
            if star_idx > 0:
                output_index.append((0, star_idx-1))
        pre_end_idx = end_idx

        output_index.append((star_idx, end_idx))

    if pre_end_idx!=max_len-1:
        output_index.append((pre_end_idx+1, max_len-1))

    return output_index
