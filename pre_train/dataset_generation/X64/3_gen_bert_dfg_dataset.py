#!/usr/bin/python
# -*- coding: UTF-8 -*-
from future.utils import viewitems
import networkx as nx
import idaapi
import idautils
import idc
import sys
import os
import time
from copy import deepcopy
from miasm.core.bin_stream_ida import bin_stream_ida
from miasm.core.asmblock import *
from miasm.core.locationdb import LocationDB
from miasm.expression.simplifications import expr_simp

from utils import guess_machine
from idautils import *
from idaapi import *
from idc import *
import re
import random
from re import compile, VERBOSE
import gc
import cxxfilt
idaapi.autoWait()

bin_num = 0
func_num = 0
function_list_file = ""
function_list_fp = None
functions=[]

curBinNum = 0


class bbls:
    id=""
    insts=[]
    define=[]
    use= []
    defuse={}
    fathernode=set()
    childnode=set()
    define=set()
    use=set()
    visited=False

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
    s = process_demangle_name(s)
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

def process_used(used, define):
    use_list = used.split(' ')
    var_use = set()

    for i in use_list:
        tmp_used = re.findall('\w', i)
        tmp_used = ''.join(tmp_used)
        if tmp_used:
            # print "used label", tmp_used
            # continue
            if tmp_used[0] in ['8']:
                tmp_used = tmp_used[1:]
            elif tmp_used[:2] in ['16', '32', '64']:
                tmp_used = tmp_used[2:]
            else:
                tmp_used = tmp_used

            if tmp_used in define:
                var_use.add(tmp_used)
            elif tmp_used[:2] == '0x':
                continue
            elif tmp_used == 'ZERO' or tmp_used == 'zero':
                continue
            elif tmp_used[:3] == 'loc':
                    continue
            elif len(tmp_used) == 1 and ord(tmp_used) in range(97,123):
                    continue
            else:
                # print "used label", tmp
                var_use.add(tmp_used)
    # print "used label!!!", var_use
    return list(var_use)


def build_dfg(DG,IR_blocks):
    IR_blocks_dfg=IR_blocks
    for in_label, in_value in IR_blocks.items():
        linenum=0

        addr = in_label
	    # Basic block structure initialization
        tempbbls = bbls()
        tempbbls.id=addr
        tempbbls.insts=[]
        tempbbls.childnode=set()
        tempbbls.fathernode=set()
        # Dict: record the left side and the right side of the equal sign
        tempbbls.defuse={}
        # Dict: record the defined variable and the used variable, the initial definition position and the final definition position
        tempbbls.defined={}
        tempbbls.used = {}
        # Set: record all defined variables in the basic block
        tempbbls.definedset = set()
        tempbbls.visited=False
        IR_blocks_dfg[addr] = tempbbls

        for i_addr, i in in_value:
            linenum+=1
            # Analyze every instruction
            tempbbls.insts.append((i_addr,i))

            if '=' not in i or "call" in i or 'IRDst' in i:
                continue

            define = i.split('=')[0].strip()
            if '[' in define:
                define=define[define.find('[')+1:define.find(']')]
            use = i.split('=')[1].strip()
            if define not in tempbbls.defined:
                tempbbls.defined[define]=[i_addr]#[linenum,0]
            else:
                tempbbls.defined[define].append(i_addr)#linenum

            if define not in IR_blocks_dfg[addr].defuse:
                IR_blocks_dfg[addr].defuse[define] = set()

            # If there are no parentheses, it is considered a simple assignment
            if '(' not in use and '[' not in use:
                IR_blocks_dfg[addr].defuse[define].add((i_addr,use))
                                
                if use not in tempbbls.used:
                    tempbbls.used[use] = [i_addr]
                else:
                    tempbbls.used[use].append(i_addr) #linenum

            #remove parentheses
            else:
                srclist = list(i)
                for i in range(len(srclist)):
                    if srclist[i] == ")" and srclist[i - 1] != ")":
                        tmp = srclist[0:i + 1][::-1]
                        for j in range(len(tmp)):
                            if tmp[j] == "(":
                                temps = "".join(srclist[i - j:i + 1])
                                if temps.count(')') == 1 and temps.count('(') == 1:
                                    temps = temps[1:-1]	 # no brackets

                                    IR_blocks_dfg[addr].defuse[define].add((i_addr,temps))
                                    if temps not in tempbbls.used:
                                        tempbbls.used[temps] = [i_addr]
                                    else:
                                        tempbbls.used[temps].append(i_addr)
                                break

                for i in range(len(srclist)):
                    if srclist[i] == "]" and srclist[i - 1] != "]":
                        tmp = srclist[0:i + 1][::-1]
                        for j in range(len(tmp)):
                            if tmp[j] == "[":
                                temps = "".join(srclist[i - j:i + 1])
                                if temps.count(']') == 1 and temps.count(']') == 1:
                                    temps = temps[1:-1]

                                    IR_blocks_dfg[addr].defuse[define].add((i_addr,temps))

                                    if temps not in tempbbls.used:
                                        tempbbls.used[temps] = [i_addr]
                                    else:
                                        tempbbls.used[temps].append(i_addr)
                                break

        # print "addr",addr
        # print "IR_blocks_dfg",IR_blocks_dfg
        #print "IR_blocks_dfg[addr].defuse",IR_blocks_dfg[addr].defuse


        # print "tempbbls.insts", tempbbls.insts
        # print "tempbbls.defined", tempbbls.defined
        # print "tempbbls.used", tempbbls.used
        
    for cfgedge in DG.edges():
        innode=str(cfgedge[0])
        outnode=str(cfgedge[1])
        # print "in out**"+innode+"**"+outnode
        if innode==outnode:
            continue
        if IR_blocks_dfg.has_key(innode):
            IR_blocks_dfg[innode].childnode.add(outnode)
        if IR_blocks_dfg.has_key(outnode):
            IR_blocks_dfg[outnode].fathernode.add(innode)

    # Find the starting node and record all variables defined in each basic block
    cfg_nodes = DG.nodes()

    startnode = None
    for addr, bbloks in IR_blocks_dfg.items():

        if len(cfg_nodes)==1 or startnode is None :
            startnode = addr
        # print addr,addr in cfg_nodes,IR_blocks_dfg[addr].fathernode
        if addr in cfg_nodes and len(IR_blocks_dfg[addr].fathernode)==0:
            startnode = addr
        for definevar in IR_blocks_dfg[addr].defuse:
            IR_blocks_dfg[addr].definedset.add(definevar)
            # print "definedset",IR_blocks_dfg[addr].definedset
    # print "IR_blocks_dfg",IR_blocks_dfg
    # print "startnode	:",startnode
    if startnode is None:
        return nx.DiGraph()
    else:
        return gen_dfg_instr_pairs(IR_blocks_dfg, startnode)


def find_var_edge(used, defined):
    find_index=set()

    i=0
    for j in range(0, len(used)):
        if defined[i]< used[j]:
            try:
                if used[j]<= defined[i+1]:
                    find_index.add((hex(defined[i]), hex(used[j])))
                elif used[j]> defined[i+1]:
                    i = i+1
                    if i < len(defined):
                        find_index.add((hex(defined[i]), hex(used[j])))
            except IndexError as e:
                find_index.add((hex(defined[i]), hex(used[j])))

    # print "find_index", find_index
    return list(find_index)
                
def gen_one_block_dfg(IR_blocks_dfg, cur_node):
    
    cur_block = IR_blocks_dfg[cur_node]
    cur_usevar = cur_block.used
    cur_definevar = cur_block.defined
    # temp_usevar = cur_usevar
    cur_edges = []

    for curvar in cur_usevar:
        if curvar in cur_definevar:
            cur_edges.extend(find_var_edge(cur_usevar[curvar],cur_definevar[curvar]))
        else:
            if len(curvar.split(' '))>1:
                temp_cur_list =  process_used(curvar, cur_definevar)
                temp_usevar = deepcopy(cur_usevar)

                for i in temp_cur_list:
                    if i not in temp_usevar:
                        temp_usevar[i]=temp_usevar[curvar]
                    else:
                        temp_usevar[i].extend(temp_usevar[curvar])
                temp_usevar.pop(curvar)
                for i in temp_cur_list:
                    if i in cur_definevar:
                        cur_edges.extend(find_var_edge(temp_usevar[i],cur_definevar[i]))
    return cur_edges


def gen_dfg_instr_pairs(IR_blocks_dfg,startnode):
        
    G = nx.DiGraph()
    G.add_node(-1, text='entry_point')

    label_dict = {}
    label_dict[-1] = 'entry_point'
        
    stack_list = []
    visited ={}
    # v2 means to visit for the second time but not visited yet
    visited2= {}
    # v3 indicates that it has been visited twice
    visited3 = {}
    for key,val in IR_blocks_dfg.items():
        visited2[key]=set()
        visited3[key]=set()
    visitorder=[]

    IR_blocks_dfg[startnode].visited=True
    visited[startnode] = '1'
    visitorder.append(startnode)
    stack_list.append(startnode)
    while len(stack_list) > 0:
        cur_node = stack_list[-1]
        next_nodes = set()

        if IR_blocks_dfg.has_key(cur_node):
            for addr, _ in IR_blocks_dfg[cur_node].insts:

                G.add_node(str(hex(addr))+'L', text=get_instruction(addr))
                label_dict[str(hex(addr))+'L'] = get_instruction(addr)
            cur_edge = gen_one_block_dfg(IR_blocks_dfg, cur_node)
            if cur_edge:
                for knode, vnode in cur_edge:
                    if not knode in G:
                        G.add_node(knode, text=get_instruction(int(knode, 16)))
                    if not vnode in G:
                        G.add_node(vnode, text=get_instruction(int(vnode, 16)))
                G.add_edges_from(cur_edge)

            next_nodes = IR_blocks_dfg[cur_node].childnode
        # print len(stack_list),cur_node,"-->",next_nodes
        if len(next_nodes) == 0:  # The leaf node needs to roll back
            stack_list.pop()
            visitorder.pop()

        else:
            if (len(set(next_nodes) - set(visited.keys())) == 0 ) and len(next_nodes & visited2[cur_node])==0:
                # If you have been visited, you need to go back
                stack_list.pop()
                visitorder.pop()

            else:
                for i in next_nodes:

                    if i not in visited or i in visited2[cur_node]:
                        fathernodes=set()
                        usevar = {}

                        if IR_blocks_dfg.has_key(i):
                            # List: parent node
                            fathernodes=IR_blocks_dfg[i].fathernode
                            # Dict: Variables used in basic blocks and where they occur
                            usevar=IR_blocks_dfg[i].used
                            # Dict: Variables defined in basic blocks and where they occur
                            definevar=IR_blocks_dfg[i].defined

                        find_undef_var = False
                        # Set: variables defined by all parent nodes
                        allfdefinevarset=set()

                        for uvar in usevar:
                            # If the used variable is not defined in the self-basic block
                            # or the used position is before the defined position,
                            # go to the parent node to find
                            father_edge = {}
                            ffather_edge = {}
                            is_in_curr = False

                            if uvar not in definevar or usevar[uvar][0] < definevar[uvar][0]:
                                un_def_var =[uvar]
                                if uvar not in definevar:
                                    if len(uvar.split(' ')) > 1:
                                                            
                                        temp_uvar_list = process_used(uvar, definevar)
                                        for temp_var_i in temp_uvar_list:
                                            if temp_var_i not in definevar or usevar[uvar][0] < definevar[temp_var_i][0]:
                                                un_def_var.append(temp_var_i)
                                                find_undef_var = True
                                            elif temp_var_i in definevar:
                                                is_in_curr = True
                                                un_def_var = un_def_var[1:]

                                    else:
                                        find_undef_var = True

                                else:
                                    find_undef_var = True
                                                        
                                if find_undef_var:
                                                            
                                    for un_var in un_def_var:

                                        for fnode in fathernodes:
                                            fdefinevarset = set()
                                            if IR_blocks_dfg.has_key(fnode):
                                                fdefinevarset = IR_blocks_dfg[fnode].definedset
                                            # print "fdefinevarset",fdefinevarset
                                            allfdefinevarset|=fdefinevarset
                                                                        
                                            if un_var in fdefinevarset:
                                                findex = IR_blocks_dfg[fnode].defined[un_var][-1]
                                                # print "findex", findex
                                                if un_var not in father_edge:
                                                    father_edge[un_var]=[(hex(findex), hex(usevar[uvar][0]))]
                                                else:
                                                    father_edge[un_var].append((hex(findex), hex(usevar[uvar][0])))
                                                # G.add_edge(findex,usevar[uvar][0])
                                                print hex(findex), '->',hex(usevar[uvar][0]),"var:",un_var
                                                #res_graph.add_edge(fnode, i)
									            #print fnode,'->',i,"var:",uvar
                                        # There may be data dependence with parent of the parent node,
                                        # and reverse search in the order of depth-first traversal
                                        for j in range(len(visitorder)-1,-1,-1):
                                            visitednode=visitorder[j]
                                            temp_definedset = set()
                                            if IR_blocks_dfg.has_key(visitednode):
                                                temp_definedset = IR_blocks_dfg[visitednode].definedset
                                            if un_var in temp_definedset - allfdefinevarset:
                                                ffindex = IR_blocks_dfg[visitednode].defined[un_var][-1]
                                                #G.add_edge(ffindex, usevar[uvar][0])
                                                if un_var not in ffather_edge:
                                                    ffather_edge[un_var]=[(hex(ffindex),hex(usevar[uvar][0]))]
                                                else:
                                                    ffather_edge[un_var].append((hex(ffindex), hex(usevar[uvar][0])))
                                                allfdefinevarset|=temp_definedset
                                                print "fffff", hex(ffindex), '->', hex(usevar[uvar][0]), "var:", un_var
                                                                
                                    if not is_in_curr:

                                        for index_flag in un_def_var[1:]:
                                            if index_flag in father_edge and un_def_var[0] in father_edge:
                                                father_edge.pop(index_flag)

                                            if index_flag in ffather_edge and un_def_var[0] in ffather_edge:
                                                ffather_edge.pop(index_flag)
                                                            
                                    final_edge=[]

                                    for _, value in father_edge.items():
                                        final_edge.extend(value)
                                    for _, value in ffather_edge.items():
                                        final_edge.extend(value)

                                    if final_edge:
                                        for knode, vnode in final_edge:
                                            if not knode in G:
                                                G.add_node(knode, text=get_instruction(int(knode,16)))
                                            if not vnode  in G:
                                                G.add_node(vnode, text=get_instruction(int(vnode,16)))
                                        G.add_edges_from(final_edge)


                        visited[i] = '1'
                        visitorder.append(i)
                        if i in visited2[cur_node]:
                            visited2[cur_node].remove(i)
                            visited3[cur_node].add(i)
                        temp_childnode = set()
                        if IR_blocks_dfg.has_key(i):
                            temp_childnode = IR_blocks_dfg[i].childnode
                        visited2[cur_node] |=(set(temp_childnode) & set(visited) )-set(visited3[cur_node])
                        stack_list.append(i)
    for node in G.nodes:
        if not G.in_degree(node):
            G.add_edge(-1, node)

    return G


def get_father_block(loc_db, blocks, cur_block, yes_keys):

    father_block = None
    for temp_block in blocks:

        if temp_block.get_next() == cur_block.loc_key:
            father_block = temp_block
    if father_block is None:
        return None
    is_Exist = False
    for yes_label in yes_keys:

        if (str(hex(loc_db.get_location_offset(father_block.loc_key))) + "L") == yes_label:
            is_Exist = True
    if not is_Exist:

        father_block = get_father_block(loc_db, blocks, father_block, yes_keys)
        return father_block
    else:

        return father_block

def rebuild_graph(loc_db, cur_block, blocks, IR_blocks, no_ir):

    yes_keys = list(IR_blocks.keys())
    no_keys = list(no_ir.keys())
    next_lable = str(hex(loc_db.get_location_offset(cur_block.loc_key))) + "L"
    father_block = get_father_block(loc_db, blocks, cur_block, yes_keys)
    if not father_block is None:
        for yes_label in yes_keys:
            if (str(hex(loc_db.get_location_offset(father_block.loc_key))) + "L")==yes_label:
                for no_label in no_keys:

                    if next_lable == no_label:
                        IR_blocks[yes_label].pop()
                        IR_blocks[yes_label].extend(IR_blocks[no_label])
                        # print "<<<del", no_label
                        # print "<<<len", len(no_ir)
                        del (no_ir[no_label])
                        del (IR_blocks[no_label])
    return IR_blocks, no_ir


def get_instruction(ea):
    return idc.GetDisasm(ea)


def dataflow_analysis(func, block_items, DG):
    global settings, lifter, ircfg

    machine = guess_machine(addr=func.start_ea)
    mn, dis_engine, lifter_model_call = machine.mn, machine.dis_engine, machine.lifter_model_call

    bs = bin_stream_ida()
    loc_db = LocationDB()

    mdis = dis_engine(bs, loc_db=loc_db, dont_dis_nulstart_bloc=True)
    lifter = lifter_model_call(loc_db)

    for ad, name in idautils.Names():
        if name is None:
            continue
        loc_db.add_location(name, ad)

    asmcfg = mdis.dis_multiblock(func.start_ea)

    ircfg = lifter.new_ircfg_from_asmcfg(asmcfg)


    IRs = {}
    for lbl, irblock in viewitems(ircfg.blocks):
        insr = []
        if loc_db.get_location_offset(lbl):

            for assignblk in irblock:
                try:
                    inst_addr = assignblk.instr.offset
                    for dst, src in viewitems(assignblk):
                        dst, src = expr_simp(dst), expr_simp(src)
                        insr.append((inst_addr, str(dst) + "=" + str(src)))
                except:
                    pass

            IRs[str(hex(loc_db.get_location_offset(lbl)))+"L"] = insr

    IR_blocks={}
    no_ir = {}

    for block in asmcfg.blocks:

        isFind = False
        item = str(hex(loc_db.get_location_offset(block.loc_key)))+ "L"

        for block_item in block_items:
            if item == block_item:
                isFind = True

        if IRs.has_key(item):
            if isFind:
                IR_blocks[item] = IRs[item]
            else:
                IR_blocks[item] = IRs[item]
                no_ir[item]= IRs[item]

    no_keys = list(no_ir.keys())
    # print "no_ir : ",no_keys
    for cur_label in no_keys:
        cur_block = None
        # print "find no_ir	 label is : ",cur_label
        for block in asmcfg.blocks:
            #remove example like loc_0000000000413D4C:0x00413d4cL callXXX
            temp_index = str(hex(loc_db.get_location_offset(block.loc_key)))+"L"
            # print block.label,temp_index
            if temp_index==cur_label:
                cur_block = block
        if not cur_block is None:
            # print "find no_ir ",cur_block
            IR_blocks, no_ir = rebuild_graph(loc_db, cur_block, asmcfg.blocks, IR_blocks, no_ir)

    IR_blocks_toDFG = {}
    for key, value in IR_blocks.items():
        if len(key.split(':'))>1:
            key = key.split(':')[0] + ":0x"+key.split(':')[1].strip()[2:].lstrip('0')
        # print "dg to dfg : ", key
        IR_blocks_toDFG[key] = value
    #print "IR_blocks_toDFG",IR_blocks_toDFG
    #print "CFG edges <<",DG.number_of_edges(),">> :",DG.edges()
    dfg = build_dfg(DG,IR_blocks_toDFG)
    #dfg.add_nodes_from(DG.nodes())
    print "CFG edges <<",DG.number_of_edges(),">> :",DG.edges()
    print "DFG edges <<",dfg.number_of_edges(),">> :",dfg.edges()
    print "DFG nodes : ",dfg.number_of_nodes(),">> :",dfg.nodes()
	
    return dfg

def filter_jump(operand):
    symbols = operand.split(' ')
    processed =[]
    for sym in symbols:
        if sym.startswith('loc_'):
            if ':' in sym:
                processed.append("LOCALFUN")
            else:
                processed.append('LOCALJUMP')
        elif sym =='short':
            processed.append('short')

        elif sym.startswith('locret_'):
            processed.append('RETJUMP')
    return processed

def filter_digit(symbols, is_neg,symbol_map, string_map):
    processed =[]

    if symbols=='0':
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
        elif len(symbols)<7:
            if is_neg:
                processed.append("NEGATIVE")
            else:
                processed.append("POSITIVE")

    elif is_hex(symbols):
        if len(symbols) >5:

            tmp_addr = int(symbols[:-1], 16)
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

        elif len(symbols)<6:
            if is_neg:
                processed.append("NEGATIVE")
            else:
                processed.append("POSITIVE")

    return processed

def get_callref(operand):
    calls = []
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

def is_hex( operand):
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
    if parts[0]=='call':
        processed = get_callref(operand)
        token_lst.extend(processed)
    else:
        for i in range(len(operand)):

            processed = []
            jump_filter = filter_jump(operand[i])
            if len(jump_filter)>0:
                processed.extend(jump_filter)
            else:
                symbols = re.split('([0-9A-Za-z_$]+)', operand[i])
                symbols = [s.strip() for s in symbols if s]

                is_neg = False
                for j in range(len(symbols)):
                    sym_digit = filter_digit(symbols[j], is_neg, symbol_map,string_map)
                    if len(sym_digit)>0:
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
                    elif symbols[j]=='-':
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

def random_walk(g, length, symbol_map, string_map):
    sequence = []
    for n in g:
        if n != -1 and g.node[n]['text'] != None:
            s = []
            l = 0
            s.append(parse_instruction(g.node[n]['text'], symbol_map, string_map))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    s.append(parse_instruction(g.node[cur]['text'], symbol_map, string_map))
                    l += 1
                else:
                    break
            sequence.append(s)
    return sequence

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
            tmp_demangle_name = tmp_demangle_name[1:]
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
    global bin_num, func_num, function_list_file, function_list_fp,functions

    if len(idc.ARGV)<1:
        print "error, please enter arguements"
    else:
        print idc.ARGV[1]
        print idc.ARGV[2]
        fea_path_origion = idc.ARGV[1]
        bin_path = idc.ARGV[2]
        bin_name = os.path.basename(bin_path)
        bin_name = bin_name.split('.')[:-1]
        if len(bin_name) > 1:
            bin_name = '-'.join(bin_name)
        else:
            bin_name = bin_name[0]
        print "bin_name", bin_name

    print "Directory path	：	", fea_path_origion

    symbol_map = {}
    string_map = {}
	
    function_graphs = {}
    for stri in idautils.Strings():
        string_map[stri.ea] = str(stri)
    for func in idautils.Functions():

        tmp_symbol_name = idc.GetFunctionName(func)
        tmp_symbol_name = process_demangle_name(tmp_symbol_name)
        symbol_map[func] =  tmp_symbol_name
    for i in range(0, get_func_qty()):
        # Ignore Library Code
        func = getn_func(i)

        segname = get_segm_name(func.start_ea)
        # get the segment name of the function by address ,x86 arch segment includes (_init _plt _plt_got _text extern _fini)
        if segname[1:3] not in ["OA", "OM", "te"]:
            continue

        cur_function_name = idc.GetFunctionName(func.start_ea)
        
        if filter_cfunction(cur_function_name):
            continue
        #cur_function_name = process_demangle_name(cur_function_name)
 

        #if cur_function_name != "X509_NAME_get_text_by_NID":
        #	 continue

        functions.append(cur_function_name.lower())
        print cur_function_name, hex(func.start_ea),"=====start"

        allblock = idaapi.FlowChart(idaapi.get_func(func.start_ea))

        block_items = []
        DG = nx.DiGraph()
        for idaBlock in allblock:
            temp_str = str(hex(idaBlock.start_ea))
            block_items.append(temp_str)
            DG.add_node(hex(idaBlock.start_ea))
            for succ_block in idaBlock.succs():
                DG.add_edge(hex(idaBlock.start_ea),hex(succ_block.start_ea))
            for pred_block in idaBlock.preds():
                DG.add_edge(hex(pred_block.start_ea),hex(idaBlock.start_ea))

        try:
            dfg = dataflow_analysis(func, block_items, DG)
            if len(dfg.nodes) > 2:
                function_graphs[cur_function_name] = dfg
        except:
            print "error!!!! error !!! {} can not analyse".format(cur_function_name)

        print cur_function_name, "=====finish"

    with open(fea_path_origion + '/' + bin_name + '_single_dfg_train.txt', 'w') as w:
        for name, graph in function_graphs.items():
            dfg_lines = []
            sequence = random_walk(graph, 40, symbol_map, string_map)
            for s in sequence:
                if len(s) >= 2:
                    for idx in range(1, len(s)):
                        dfg_lines.append(s[idx - 1] + '\t' + s[idx])
                        print(dfg_lines.append(s[idx - 1] + '\t' + s[idx]))
            w.write(name + ';' + ';'.join(dfg_lines) + '\n')
    gc.collect()
                
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

