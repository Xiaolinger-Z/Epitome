#!/usr/bin/python
# -*- coding: UTF-8 -*-
import networkx as nx
import idaapi
import idautils
import idc
import sys
import os
import time
import re
import random
from idautils import *
from idaapi import *
from idc import *
from re import compile, VERBOSE
import cxxfilt

idaapi.autoWait()

bin_num = 0
func_num = 0
function_list_file = ""
function_list_fp = None
functions=[]

curBinNum = 0

RE_WORDS = compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+  # Numbers
    ''', VERBOSE)


RE_WORDS_FORMATSTR = compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ | # All upper case
    %\d?\.?\d?s | #format str s
    %\d?\.?\d?d | #format str d
    %\d?\.?\d?f | #format str f  (s|d|f makes findall go crazy :|)
    \-\d+ | # neg Numbers
    \d+ # Numbers
    ''', VERBOSE)

fmt_str = compile(r'%(\d+(\.\d+)?)?(d|f|s)')

def get_split_subtokens_proc_name(s):
    # get sub-tokens. Remove them if len()<1 (single letter or digit)
    return [x for x in [str(x).lower() for x in RE_WORDS.findall(s)] if len(x) > 1]

def get_split_subtokens_global_str(s):
    final_out = []
    tmp_str = s.split(':')
    for s_i in tmp_str:
        s_i = process_demangle_name(s_i)
        s_i_out = [x for x in [str(x).lower() for x in RE_WORDS_FORMATSTR.findall(s_i)] if fmt_str.match(x) is None if
                   len(x) > 1]
        final_out.extend(s_i_out)
    return final_out


def filter_jump(operand):
    symbols = operand.split(' ')
    processed = []
    for sym in symbols:
        if sym.startswith('loc_'):
            if ':' in sym:
                processed.append("LOCALFUN")
            else:
                processed.append('LOCALJUMP')
        elif sym == 'short':
            processed.append('short')

        elif sym.startswith('locret_'):
            processed.append('RETJUMP')
    return processed


def filter_digit(symbols, is_neg, symbol_map, string_map):
    processed = []

    if symbols == '0':
        processed.append('ZERO')
        return processed
    if symbols.startswith('dword_'):
        processed.append('address')
        return processed

    if symbols.isdigit():
        if is_neg:
            processed.append("NEGATIVE")
        else:
            processed.append("POSITIVE")
    elif symbols[:2] == '0x':
        if len(symbols) > 6 and len(symbols) < 15:

            tmp_addr = int(symbols, 16)
            if tmp_addr in symbol_map:
                tmp_sym = symbol_map[tmp_addr]

                processed.extend(get_split_subtokens_global_str(tmp_sym))
            elif tmp_addr in string_map:
                tmp_sym = string_map[tmp_addr]

                processed.extend(get_split_subtokens_global_str(tmp_sym))
            else:
                if is_neg:
                    processed.append("negaddress")
                else:
                    processed.append("address")
        elif len(symbols) < 7:
            if is_neg:
                processed.append("NEGATIVE")
            else:
                processed.append("POSITIVE")

    elif is_hex(symbols):
        if len(symbols) > 5:

            tmp_addr = int(symbols[:-1], 16)
            if tmp_addr in symbol_map:

                tmp_sym = symbol_map[tmp_addr]

                processed.extend(get_split_subtokens_global_str(tmp_sym))
            elif tmp_addr in string_map:
                # print "string ####### tmp_addr"
                tmp_sym = string_map[tmp_addr]

                processed.extend(get_split_subtokens_global_str(tmp_sym))
            else:
                if is_neg:
                    processed.append("negaddress")
                else:
                    processed.append("address")

        elif len(symbols) < 6:
            if is_neg:
                processed.append("NEGATIVE")
            else:
                processed.append("POSITIVE")

    return processed


def get_callref(operand):
    calls = []
    print('call', operand)
    for i in range(len(operand)):

        if operand[i].startswith('sub'):
            calls.append('ICALL')
        elif operand[i].startswith('cs:'):
            calls.append('ICALL')
        elif 'qword' in operand[i]:
            calls.append('ICALL')
        elif '$' in operand[i]:
            calls.append('ICALL')
        elif len(operand[i]) > 3:
            calls.append('ECALL')
            tmp_operand = get_split_subtokens_global_str(operand[i])

            calls.extend(tmp_operand)
        else:
            calls.append('ICALL')

    return calls


def is_hex(operand):
    pattern = re.compile(r'^[A-F0-9]+$')

    if operand.endswith("h"):
        if pattern.match(operand[:-1]):
            return True
        else:
            return False
    else:
        return False


def parse_instruction(ins, symbol_map, string_map):

    ins = ins.split(';')[0]
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    token_lst = []
    if len(parts) > 1:
        operand = parts[1:]
    token_lst.append(parts[0])
    if parts[0] == 'call':
        processed = get_callref(operand)
        token_lst.extend(processed)
    else:
        for i in range(len(operand)):

            processed = []
            jump_filter = filter_jump(operand[i])
            if len(jump_filter) > 0:
                processed.extend(jump_filter)
            else:
                symbols = re.split('([0-9A-Za-z_$]+)', operand[i])
                symbols = [s.strip() for s in symbols if s]

                is_neg = False
                for j in range(len(symbols)):
                    sym_digit = filter_digit(symbols[j], is_neg, symbol_map, string_map)
                    if len(sym_digit) > 0:
                        processed.extend(sym_digit)
                        is_neg = False
                    elif 'var_' in symbols[j]:
                        if is_neg:
                            processed.pop()
                            processed.append("-")
                            processed.append("LOCALVAR")
                            is_neg = False
                        else:
                            processed.append("LOCALVAR")
                    elif symbols[j] == '-':
                        is_neg = True
                        processed.append("+")
                    elif len(symbols[j]) < 5:
                        if is_neg:
                            processed.pop()
                            processed.append("-")
                            processed.append(symbols[j])
                            is_neg = False
                        else:
                            processed.append(symbols[j])
                    else:
                        processed_name = get_split_subtokens_global_str(symbols[j])
                        if is_neg:
                            processed.pop()
                            processed.append("-")
                            processed.extend(processed_name)
                            is_neg = False
                        else:
                            processed.extend(processed_name)

                    processed = [p for p in processed if p]

            token_lst.extend(processed)

    return ' '.join(token_lst)

def random_walk(g,length, symbol_map, string_map):
    sequence = []
    for n in g:
        if n != -1 and 'text' in g.node[n]:
            s = []
            l = 0
            s.append(parse_instruction(g.node[n]['text'], symbol_map, string_map))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    if 'text' in g.node[cur]:
                        s.append(parse_instruction(g.node[cur]['text'], symbol_map, string_map))
                        l += 1
                    else:
                        break
                else:
                    break
            sequence.append(s)
        if len(sequence) > 5000:
            print("early stop")
            return sequence[:5000]
    return sequence

def get_instruction(ea):
    return idc.GetDisasm(ea)

def build_one_arch_cfg (func):
    
    allblock = idaapi.FlowChart(idaapi.get_func(func.start_ea))
    G = nx.DiGraph()

    for idaBlock in allblock:
        curEA = idaBlock.start_ea

        predecessor = hex(curEA)

        while curEA <= idaBlock.end_ea:

            G.add_node(hex(curEA), text=get_instruction(curEA))
            if curEA != idaBlock.start_ea:
                G.add_edge(hex(predecessor), hex(curEA))
            predecessor = curEA
            curEA = idc.NextHead(curEA,idaBlock.end_ea)
        
        for succ_block in idaBlock.succs():
            G.add_edge(hex(predecessor), hex(succ_block.start_ea))
        for pred_block in idaBlock.preds():
            G.add_edge(hex(pred_block.end_ea), hex(idaBlock.start_ea))

    return G

class Stack:
    def __init__(self):
        self.lista = []

    def isEmpty(self):
        return len(self.lista) == 0
    def push(self,item):
        self.lista.append(item)
    def pop(self):
        if self.isEmpty():
            return "Error：The stack is empty"
        else:
            return self.lista.pop()


def matching(inputstring, leftbkt, rightbkt):
    bktStack = Stack()
    #leftbkt = "{[(<"
    #rightbkt = "}])>"

    tmp_str=[]
    is_print=True
    for index, i in enumerate(inputstring):
        if i in leftbkt:

                bktStack.push(i)
                is_print = False
        elif i in rightbkt:
            if bktStack.isEmpty():
                is_print = True
            elif rightbkt.index(i) ==  leftbkt.index(bktStack.pop()):
                if bktStack.isEmpty():
                    is_print = True
                    continue
        if is_print:
            tmp_str.append(i)

    return ''.join(tmp_str)

def tmp_process( tmp_name):

    try:
        demangle_name = cxxfilt.demangle(re.split(r'[.@]', tmp_name)[0])
    except:
        # if '.' in tmp_demangle_name:
        #    tmp_demangel_name = tmp_demangle_name.split('.')[0]
        if '_isra_'in tmp_name:
            i_index = [i.start() for i in re.finditer('_isra_', tmp_name)]
            tmp_name = tmp_name[:i_index[0]]

        if '_part_' in tmp_name:
            i_index = [i.start() for i in re.finditer('_part_', tmp_name)]
            tmp_name = tmp_name[:i_index[0]]

        if '_constprop_'in tmp_name:
            i_index = [i.start() for i in re.finditer('_constprop_', tmp_name)]
            tmp_name = tmp_name[:i_index[0]]

        if '_' in tmp_name[1:]:
            # print(tmp_demangle_name.split('_')[:-1])
            tmp_de_name = tmp_name.split('_')
            try :
                tmp_demangle_name = '_'.join(tmp_de_name)
                demangle_name = cxxfilt.demangle(tmp_demangle_name)
            except:
                try:
                    tmp_demangle_name = '_'.join(tmp_de_name[:-1])
                    demangle_name = cxxfilt.demangle(tmp_demangle_name)
                except:
                    try:
                        tmp_demangle_name = '_'.join(tmp_de_name[:-2])
                        demangle_name = cxxfilt.demangle(tmp_demangle_name)
                    except:
                        demangle_name = '_'.join(tmp_de_name)
        else:
            try:
                demangle_name = cxxfilt.demangle(tmp_name)
            except:
                demangle_name = tmp_name

    if '(' in demangle_name:
        demangle_name = matching(demangle_name, '(', ')')
    if '<' in demangle_name:
        demangle_name = matching(demangle_name, '<', '>')

    return demangle_name

def process_demangle_name(tmp_demangle_name):

    if tmp_demangle_name.startswith('_Z'):
        demangle_name = tmp_process(tmp_demangle_name)

    elif '_Z' in tmp_demangle_name[:3]:
        if tmp_demangle_name[0].isalnum():
            demangle_name = tmp_demangle_name
        else:
            tmp_demangle_name=tmp_demangle_name[1:]
            demangle_name = tmp_process(tmp_demangle_name)

    else:
        demangle_name = tmp_demangle_name

    return demangle_name

def filter_cfunction(tmp_demangle_name):

    if tmp_demangle_name.startswith('_Z'):
        return True

    elif '_Z' in tmp_demangle_name[:3]:
        if tmp_demangle_name[0].isalnum():
            return False
        else:
            return True
    elif tmp_demangle_name.startswith('sub_'):
        return True
    elif tmp_demangle_name =='main':
        return True
    elif tmp_demangle_name.startswith('_GLOBAL_'):
        return True
    elif tmp_demangle_name.endswith('.cc'):
        return True
    elif tmp_demangle_name.endswith('.cpp'):
        return True
    else:
        return False

def main():

    global bin_num, func_num, function_list_file, function_list_fp, functions

    if len(idc.ARGV) < 1:
        print "error, please enter arguements"

    else:
        print idc.ARGV[1]
        print idc.ARGV[2]
        fea_path_origion = idc.ARGV[1]
        bin_path = idc.ARGV[2]

        bin_name2 = os.path.basename(bin_path)
        bin_name = bin_name2.split('.')[:-1]
        if len(bin_name)>1:
            bin_name = '-'.join(bin_name)
        else:
            bin_name = bin_name[0]

        print "bin_name", bin_name

    print "Directory path	：	", fea_path_origion

    symbol_map = {}
    string_map = {}
    single_function_graphs = {}
    single_function_raw = {}

    for stri in idautils.Strings():
        string_map[stri.ea] = str(stri)

    for func in idautils.Functions():

        tmp_symbol_name = idc.GetFunctionName(func)
        tmp_symbol_name = process_demangle_name(tmp_symbol_name)
        symbol_map[func] = tmp_symbol_name

    for i in range(0, get_func_qty()):
        # Ignore Library Code

        func = getn_func(i)
        segname = get_segm_name(func.start_ea)

        if segname[1:3] not in ["OA", "OM", "te"]:
            continue

        cur_function_name = idc.GetFunctionName(func.start_ea)

        if filter_cfunction(cur_function_name):
            continue


        print cur_function_name, hex(func.start_ea), "=====start"

        single_graph = build_one_arch_cfg(func)

        allblock = idaapi.FlowChart(idaapi.get_func(func.start_ea))
        label_dict = {}

        for idaBlock in allblock:
            curEA = idaBlock.start_ea

            while curEA <= idaBlock.end_ea:
                # print "inst addr {} {}".format(curEA, get_instruction(curEA))
                label_dict[hex(curEA)] = get_instruction(curEA)
                curEA = idc.NextHead(curEA, idaBlock.end_ea)

        if len(single_graph.nodes) > 2:
            single_function_graphs[cur_function_name] = single_graph
            single_function_raw[cur_function_name] = label_dict

        print cur_function_name, "=====finish"

    # Write single arch CFG in txt (Assembly instruction coding)
    cfg_A = open(fea_path_origion+'/'+ bin_name+'_cfg_A.txt', 'w')
    cfg_graph_indict = open(fea_path_origion + '/' + bin_name+'_cfg_graph_indicator.txt', 'w')
    cfg_graph_label = open(fea_path_origion + '/raw_' + bin_name +'_cfg_graph_labels.txt', 'w')
    cfg_node_attri = open(fea_path_origion + '/' + bin_name +'_cfg_node_attributes.txt', 'w')
    raw_inst = open(fea_path_origion + '/' + bin_name + '_inst.txt', 'w')
    raw_label = open(fea_path_origion + '/raw_' + bin_name + '_labels.txt', 'w')
    
    # clear the file
    '''
    cfg_A.wirte('')
    cfg_graph_indict.write('')
    cfg_graph_label.write('')
    cfg_node_attri.write('')
    raw_inst.write('')
    raw_label.write('')
    '''

    node_total = 0
    for graph_index, (name, graph) in enumerate(single_function_graphs.items()):

        graph_tm = nx.convert_node_labels_to_integers(graph,first_label=1)
        # print "after:\n", graph_tm.node[0]['text']
        # Because python2.X and Python 3.X syntax are incompatible, a other script is used to process function names

        cfg_graph_label.write(name +'\n')
        raw_label.write(name + '\n')

        for node in graph_tm.nodes():

            cfg_graph_indict.write(str(graph_index+1)+'\n')
            node_raw_label = graph_tm.node[node]['text']
            node_arr = parse_instruction(node_raw_label, symbol_map, string_map)

            cfg_node_attri.write(node_arr+'\n')
            for edge  in graph_tm.succ[node]:
                #cfg_str = node + "," + edge
                cfg_A.write(str(node+node_total) + "," + str(edge+node_total) + '\n')
                cfg_A.write(str(edge+node_total) + "," + str(node+node_total) + '\n')

        node_total+= len(graph_tm.nodes())

        raw_data = single_function_raw[name]

        for tmp_inst in raw_data.values():
            inst_arr = parse_instruction(tmp_inst, symbol_map, string_map)

            raw_inst.write(inst_arr+'\t')
        raw_inst.write('\n')
    
    cfg_A.close()
    cfg_graph_indict.close()
    cfg_graph_label.close()
    cfg_node_attri.close()
    raw_inst.close()
    raw_label.close()

    return

#redirect output into a file, original output is the console.
def stdout_to_file(output_file_name, output_dir=None):
    if not output_dir:
        output_dir = os.path.dirname(os.path.realpath(__file__))
    output_file_path = os.path.join(output_dir, output_file_name)
    print output_file_path
    print "original output start"
    # save original stdout descriptor
    orig_stdout = sys.stdout
    # create output file
    f = file(output_file_path, "w")
    # set stdout to output file descriptor
    sys.stdout = f
    return f, orig_stdout


if __name__=='__main__':
	f, orig_stdout = stdout_to_file("output_"+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+".txt")
	main()
	print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
	sys.stdout = orig_stdout #recover the output to the console window
	f.close()

	idc.Exit(0)
