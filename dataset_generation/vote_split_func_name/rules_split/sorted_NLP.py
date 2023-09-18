import os, re, sys
from re import compile, VERBOSE
import enchant, functools, itertools
import nltk, math, copy
from nltk.corpus import wordnet as wn
from threading import Lock
from intervaltree import Interval, IntervalTree
import numpy as np
import itertools
from .utils import find_abbr_index, sorted_label_token

RE_WORDS = compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+  # Numbers
    ''', VERBOSE)

class NLP:

    def __init__(self, config):
        self.config = config
        self.enchant_lock = Lock()
        
        file_path = "split_name_word.txt"
        self.program_D = enchant.request_pwl_dict(file_path)
        #self.us_D = enchant.DictWithPWL("en_US", file_path)#enchant.Dict("en_US")
        self.us_D = enchant.Dict("en_US")
        self.gb_D = enchant.Dict("en_GB")

        #regexes
        self.underprefix    = re.compile(r'^_+')
        self.undersuffix    = re.compile(r'_+$')
        self.bitssuffix     = re.compile(r'(32|64)$')
        self.bitsprefix     = re.compile(r'^(32|64)')

        self.r2_prefix      = re.compile(r'^sym\.')
        self.r2_dyn_prefix  = re.compile(r'^sym\.imp\.')

        self.isra           = re.compile(r'\.isra(\.\d*)*')
        self.part           = re.compile(r'\.part(\.\d*)*')
        self.constprop      = re.compile(r'\.constprop(\.\d*)*')
        self.constp         = re.compile(r'\.constp(\.\d*)*')

        #self.libc           = re.compile(r'libc\d*')
        self.sse2           = re.compile(r'_sse\d*')
        self.ssse3          = re.compile(r'_ssse\d*')
        self.avx            = re.compile(r'avx\d*')
        self.cold           = re.compile(r'\.cold$')

        self.unaligned      = re.compile(r'unaligned')
        #self.internal = re.compile(r'internal')
        self.erms           = re.compile(r'erms')
        self.__ID__         = re.compile(r'_+[A-Z]{1,2}_+')

        self.dot_prefix     = re.compile(r'^\.+')
        self.dot_suffix     = re.compile(r'\.+$')
        self.num_suffix     = re.compile(r'_+\d+$')
        self.num_prefix     = re.compile(r'^\d+_+')
        self.dot_num_suffix = re.compile(r'\.+\d+$')
        self.num_only_prefix = re.compile(r'^\d+')
        self.num_only_suffix = re.compile(r'\d+$')

        self.repeated_nonalpha = re.compile(r'([^a-zA-Z0-9\d])\1+')

        self.ida_import     = re.compile(r'__imp_')
        self.data_lib       = re.compile(r'@@.*')

        self.abbreviations = {
            'mem' : ['member','memory'],
            'char' : 'character',
            'arg' : 'argument',
            'att': 'attribute',
            'attr': 'attribute',
            'a11y' : 'accessibility',
            'algo' : 'algorithm',
            'arp': 'arp',
            'cmp' : 'compare',
            'cpy' : 'copy',
            'str' : 'string',
            'mov' : 'move',
            'ins' : 'insert',
            'init' : ['initialize', 'initiator'],
            'ini' : ['initialize', 'initiator'],
            'deinit': 'remove_initialize',
            'fini' : 'finalize',
            'dup': 'duplicate',
            'tup': 'tuple',
            'up': 'update',
            #'va' : 'various_arguments',
            'msg' : 'message',
            'ts' : 'timestamp',
            'intel': 'computer_architecture',
            'int' : 'integer',
            'buf' : 'buffer',
            'buff' : 'buffer',
            'sep' : 'separate',
            'gcc' : 'compiler',
            'clang' : 'compiler',
            'db' : 'database',
            'dblogs': 'db_logs',
            'cb' : 'callback',
            'hw' : 'hardware',
            'profil' : 'profile',
            'contrl':'control',
            #'con' : 'connect',
            'conn' : 'connect',
            'const':'const',
            'sig' : 'signal',
            'jmp' : 'jump',
            'proc' : 'process',
            'eval' : 'evaluate',
            'vas': 'vas',
            'gen': ['generate', 'generic'],
            'abrt' : 'abort',
            'alloc' : 'allocate',
            'malloc' : 'memory_allocate',
            'sys' : 'system',
            'io' : 'input_output',
            'fcn' : 'function',
            'fn' : 'function',
            'func': 'function',
            'nan' : 'nan',
            'opt' : 'option',
            'aux' : 'auxiliary',
            'mul' : 'multiply',
            'div' : 'divide',
            'subj': 'subject',
            'sub' : ['subtract', 'subject'],
            'comerr':'com_error',
            'printerr':'print_error',
            'print':'printf',
            'scan':'scanf',
            'add' : 'addition',
            #'ext': ' extent',
            #'exp' : 'exponential',
            'lex': ['lex', 'lexical'],
            'rotate': 'rotate',
            'loc' : 'location',
            'pid' : 'process_identifier',
            'gid' : 'group_identifier',
            'guid': 'globally_unique_identifier',
            'uid' : 'user_identifier',
            'uuid': 'universally_unique_identifier',
            'egid' : 'effective_group_identifier',
            'euid' : 'effective_user_identifier',
            'sgid' : 'set_group_identifier',
            'suid' : 'set_user_identifier',
            'oid': 'object_identifier',
            'id': 'identifier',
            'iter' : 'iterate',
            'err' : 'error',
            'stp' : 'string_pointer',
            'len' : 'length',
            'pad' : 'padding',
            'delim' : 'delimiter',
            'sched' : 'schedule',
            'info' : 'information',
            'stdin': 'standard_in',
            'std' : 'standard',
            'rtn' : 'return',
            'rcv' : 'receive',
            #'ip' : 'internet_protocol',
            'reg' : 'register',
            'stat' : ['state','status'],
            'dir' : ['dirent','directory'],
            'mmap' : 'memory_map',
            'punct' : 'punctuation',
            'resolv' :'resolve',
            'lst' : 'list',
            'res' : ['restore','result','resource','research'],
            'search':'search',
            'munge':'munge',
            'eq' : 'equal',
            'conv' : 'convert',
            'async' : 'asynchronous',
            'sync' : 'synchronous',
            'fd' : 'file_descriptor',
            'alnum' : 'alpha_numeric',
            'avg' : 'average',
            'cwd' : 'current_working_directory',
            'pwd' : 'print_working_directory',
            'lib' : 'library',
            'conf' : 'configuration',
            'os' : 'operating_system',
            'chr' : 'character',
            'src' : 'source',
            'dst' : 'destination',
            'dest' : ['destroy','destination'],
            'tow' : 'to_wide',
            'dl' : 'dynamically_linked',
            'tty' : 'terminal',
            'pts' : 'pseudo_terminal',
            'cspn' : 'character_span',
            'dents' : 'directory_entries',
            'tz' : 'time_zone',
            'wc' : 'wide_character',
            'gz' : 'gzip',
            'mk': 'make',
            'rm' : 'remove',
            'param': 'parameter',
            'seg': 'segment',
            'bool' :'boolean',
            'tmp': 'temp',
            'lck': 'lock',
            'unlck':'unlock',
            #'mkdir' :'make_directory',
            #'rmdir': 'remove_directory',
            'fsck' : 'file_system_check',
            'nat' :'network_address_translation',
            'toa' : 'to_ascii',
            'pos' : ['post', 'position'],
            'chk' : 'check',
            'expr' : 'expression',
            'ind' : 'index',
            'errno' : 'error',
            'assert' : 'assertion',
            'addr' : 'address',
            'mux': 'mux',
            'ux' : 'user_interface',
            'p2p': 'peer_to_peer',
            'tbl':'table',
            'settime':'set_time',
            'tgt':'target',
            'der' : ['deregister','distinguished_encoding_rules'],
            'ber' : 'basic_encoding_rules',
            'rdn' : 'relative_distinguished_name',
            'clearerr': 'clear_error',
            'altsetting': 'alternative_setting',
            'gethostbyname': 'get_host_by_name',
            'gethostbyaddr' : 'get_host_by_address',
            'timesend': 'time_send',
            'timeset': 'time_set',
            'timestamp': 'time_stamp',
            'timediff': 'time_diff',
            'tildexpand': 'tilde_expand',
            'makesuffix': 'make_suffix',
            'coldef':'col_def',
            'checksig':'check_sig',
            'env':'environment',
            'privkey': 'privkey',
            'pubkey': 'pubkey',
            'cksum': 'check_sum',
            'wdb': 'wind_debug',
            'usr': 'user',
            'creat': 'create',
            'sock': 'socket',
            'setsockopt': 'set_socket_option',
            'clust':'cluster',
            'soinsert': 'so_insert',
            'que': ['qusetion', 'queue', 'query'],
            #missing definitions
            'cvt': 'convert',
            'num': 'number',
            'cond':['condition', 'conduct'],
            'hook' : 'hook',
            'to' : 'to',
            'i18n' : 'i18n',
            'posix' : 'posix',
            'amd' : 'computer_architecture',
            'unmap' : 'unmap',
            'free' : 'free',
            'dis': 'dis',
            'iso' : 'iso',
            'is' : ['iso','is'],
            'at' : 'at',
            'align' : 'align',
            'open' : 'open',
            'utf-8' : 'utf-8',
            'utf-16' : 'utf-16',
            'ascii' : 'ascii',
            'libc': 'libc',
            'ipv': 'ipv',
            'ip':['ipc', 'ipv', 'ip'],
            'by': 'by',
            'ios': 'ios',
            'part':['partitioned','partition','partial','particular','part'],
            'send':['send', 'sender'],
            #'fsck':'fsck',
            'mcm' :'mcm',
            'mcmp' : 'mcmp',
            'mcs': 'mcs',
            'bifs': 'bifs',
            'bif': 'bifs',
            'log':'log',
            'blk':'blk',
            'redo':'redo',
            'dmap':'dmap',
            'dma': 'direct_memory_access',
            'vmod':'vmod',
            'flca':'flca',
            'dopen':'dopen',
            'block':'block',
            'raise':'raise',
            'seterrno':'set_error',
            'dev':'dev',
            'complist': 'complist',
            'com':['command', 'compute', 'computer', 'commit','com'],
            'cmd':'cmd',
            'md':'md5',
            'iser':'iser',
            'ibus':'ibus',
            'dbus':'dbus',
            'unix':'unix',
            'java':'java'
        }

        self.expand_abbr = {
            'mem': 'memory',
            'char': 'character',
            'arg': 'argument',
            'att': 'attribute',
            'attr': 'attribute',
            'a11y': 'accessibility',
            'algo': 'algorithm',
            'cmp': 'compare',
            'cpy': 'copy',
            'str': 'string',
            'mov': 'move',
            'ins': 'insert',
            'init': 'initialize',
            'ini': 'initialize',
            'deinit': 'remove_initialize',
            'fini': 'finalize',
            'va' : 'various_arguments',
            'msg': 'message',
            'ts': 'timestamp',
            'intel': 'computer_architecture',
            'int': 'integer', #['inter','integer'],
            'buf': 'buffer',
            'buff': 'buffer',
            'sep': 'separate',
            'gcc': 'compiler',
            'clang': 'compiler',
            'wdb': 'wind_debug',
            'db': 'database',
            'dblogs': 'db_logs',
            'cb': 'callback',
            'hw': 'hardware',
            'profil': 'profile',
            'contrl': 'control',
            # 'con' : 'connect',
            'conn': 'connect',
            'cat': 'catalog',
            'sig': 'signal',
            'jmp': 'jump',
            'proc': 'process',
            'eval': 'evaluate',
            'gen': 'generate',
            'abrt': 'abort',
            'alloc': 'allocate',
            'malloc': 'memory_allocate',
            'sys': 'system',
            'io': 'io',
            'dup': 'duplicate',
            'fcn': 'function',
            'fn': 'function',
            'func': 'function',
            'na': 'nan',
            'opt': 'option',
            'aux': 'auxiliary',
            'mul': 'multiply',
            'div': 'divide',
            'subj': 'subject',
            'seg': 'segment',
            # 'exp' : 'exponential',
            'loc': 'location',
            'id': 'identifier',
            'iter': 'iterate',
            'err': 'error',
            'stp': 'string_pointer',
            'len': 'length',
            'delim': 'delimiter',
            'sched': 'schedule',
            'info': 'information',
            'stdin': 'standard_in',
            'std': 'standard',
            'tmp': 'temp',
            'cksum':'check_sum',
            'resolv' :'resolve',
            'lst' : 'list',
            'que': 'queue',
            'lck': 'lock',
            'unlck':'unlock',

            # 'ip' : 'internet_protocol',
            'reg': 'register',
            'stat': 'status',
            'dir': 'directory',
            'mmap': 'memory_map',
            'punct': 'punctuation',
            'res': 'resource',
            'eq': 'equal',
            'conv': 'convert',
            'async': 'asynchronous',
            'sync': 'synchronous',
            'fd': 'file_descriptor',
            'alnum': 'alpha_numeric',
            'avg': 'average',
            'cwd': 'current_working_directory',
            'pwd': 'print_working_directory',
            'lib': 'library',
            'conf': 'configuration',
            'os': 'operating_system',
            'chr': 'character',
            'src': 'source',
            'dst': 'destination',
            'dest': 'destination',
            'tow': 'to_wide',
            'dl': 'dynamically_linked',
            'tty': 'terminal',
            'pts': 'pseudo_terminal',
            'cspn': 'character_span',
            'dents': 'directory_entries',
            'tz': 'time_zone',
            'wc': 'wide_character',
            'gz': 'gzip',
            'mk': 'make',
            'rm': 'remove',
            'param': 'parameter',
            'rtn' : 'return',
            'rcv' : 'receive',
            # 'mkdir' :'make_directory',
            # 'rmdir': 'remove_directory',
            'fsck': 'file_system_check',
            'nat': 'network_address_translation',
            'toa': 'to_ascii',
            'pos': 'position',
            'chk': 'check',
            'expr': 'expression',
            'ind': 'index',
            'errno': 'error',
            'assert': 'assertion',
            'addr': 'address',
            'ux': 'user_interface',
            'p2p': 'peer_to_peer',
            'tbl': 'table',
            'settime': 'set_time',
            'tgt': 'target',
            'der': 'distinguished_encoding_rules',
            'ber': 'basic_encoding_rules',
            'rdn': 'relative_distinguished_name',
            'clearerr': 'clear_error',
            'altsetting': 'alternative_setting',
            'env': 'environment',
            'usr': 'user',
            'creat': 'create',
            'sock': 'socket',
            'setsockopt': 'set_socket_option',
            'clust':'cluster',
            'cvt':'convert',
            'num': 'number',
            'cond': 'condition', #['condition', 'conduct'],
            # missing definitions
            'mcmp': 'compare',
            'hook': 'hook',
            'to': 'to',
            'i18n': 'i18n',
            'posix': 'posix',
            'amd': 'computer_architecture',
            'dma': 'direct_memory_access',
            'seterrno': 'set_error',
            'md': 'md5',
            'fat': 'fat16',
        }

        self.expand_split_abbrs = {
            'deinit': 'de init',
            'wdb': 'w db',
            'tildexpand': 'tild expand',
            'timesend' : 'time send',
            'timeset': 'time set',
            'timestamp': 'time stamp',
            'makesuffix': 'make suffix',
            'timediff': 'time diff',
            'dblogs': 'db logs',
            'malloc': 'm alloc',
            'settime': 'set time',
            'clearerr': 'clear err',
            'altsetting': 'alt setting',
            'setsockopt': 'set sock opt',
            'seterrno': 'set err no',
            'coldef': 'col def',
            'checksig': 'check sig',
        }


        self.stopwords =['my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'the', 'and', 'but', 'if', 'or', 'because', 'until', 'while', 'of', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    def expand_abbreviations(self, abbr):
        if abbr in self.expand_abbr:
            return self.expand_abbr[abbr]
        return abbr

    def _score_abbrs(self, name, abbrs):
        """
            Score a permutation of abbrs in a string
        """
        t = IntervalTree()
        score = 0
        used = set([])

        for abbr in abbrs:
            start = name.index(abbr)
            end = start + len(abbr)
            if t.overlaps(start, end):
                continue
            t.addi(start, end, len(abbr))
            score += (len(abbr) ** 2) / 2
            used.add(abbr)

        return score, used

    def subtract_words_sequence(self, name, abbrs):
        """
            Return a set of words that are split by known abbreviations
            e.g. awdddwordkjh, [ word ] -> [awddd, word, kjh]
        """
        assert(isinstance(abbrs, list))
        words = []
        splits = set([name])
        acc = 0
        for word in abbrs:
            #print(word)
            #print(splits)
            #print(words)
            new_splits = set([])
            for split in splits:
                try:
                    s = split.index(word)
                    words.append(split[s:s+len(word)])
                    if s > 0:
                        new_splits.add(split[:s])
                    if s + len(word) < len(split):
                        new_splits.add(split[s+len(word):])

                    #found, continue
                    splits = copy.deepcopy(new_splits)
                    break

                except ValueError:
                    new_splits.add(split)
                    continue

        return words + list(splits)

    def _combinations_with_condition(self, l, cond_value, base=[]):
        #recursively generate all combinations that meet conditions
        assert(isinstance(l, list))
        max_dimensions = len(l)

        for i in range(len(base), max_dimensions):
            if l[i] in base:
                continue
            new_comb = base + [ l[i] ]
            size = functools.reduce(lambda x, y: x + len(y), new_comb, 0)
            if size <= cond_value:
                yield new_comb
            else:
                break

            yield from self._combinations_with_condition(l, cond_value, base=copy.deepcopy(new_comb))

    def best_cut_of_the_rod(self, name, abbrs):
        """
            return the best subset that maximizes the know characters in name
        """
        #perms = itertools.permutations(abbrs)
        #perms = itertools.combinations(abbrs)
        combs = self._combinations_with_condition(list(abbrs), len(name))
        bscore = -1
        bset = set([])
        for comb in combs:
            score, subabbrs = self._score_abbrs(name, comb)
            if score > bscore:
                bscore = score
                bset = subabbrs
                #max score
                if score == (len(name) ** 2) / 2:
                    return bset
        return bset

    def find_long_word(self, alpha_chars, base, bot, top):
        words =[]
        index = bot
        for ind in range(bot, top):
            self.enchant_lock.acquire()
            if self.program_D.check(alpha_chars[base:ind]):
                self.enchant_lock.release()
                words.append(alpha_chars[base:ind])
                # tmp_word.append(([alpha_chars[bot:ind]], ind))
                index = ind
                break
            elif self.us_D.check(alpha_chars[base:ind]) or self.gb_D.check(alpha_chars[base:ind]):
                self.enchant_lock.release()

                words.append(alpha_chars[base:ind])
                # tmp_word.append(([alpha_chars[bot:ind]], ind))
                index = ind
                break

            self.enchant_lock.release()
        return words, index
    
    def find_abbr_in_word(self, abbr, full_abbr, alpha_chars):
        words=[]
        if abbr in alpha_chars:
            if full_abbr in alpha_chars:
                words.append(full_abbr)
                alpha_chars = alpha_chars[:alpha_chars.index(full_abbr)] + \
                              alpha_chars[alpha_chars.index(full_abbr) + len(full_abbr):]

        return words, alpha_chars


    def quick_subabbreviations(self, alpha_chars):
        """
            convert strcmp -> [string, compare]
            getlanguagespecificdata -> [get, language, specific, data]
        """

        MIN_WORD_LEN = 3
        MIN_ABBR_LEN = 2
        words = []

        if self.program_D.check(alpha_chars) or self.us_D.check(alpha_chars) or self.gb_D.check(alpha_chars):
            return [ alpha_chars ]


        ##find abbreviations in words
        ##find abbreviations in words
        for abbr, full_abbr in self.abbreviations.items():
            if len(abbr) <= 3:
                continue

            tmp_word = []

            is_find_full_abbr = False
            if type(full_abbr) == list:
                for tmp_abbr in full_abbr:
                    tmp_word, alpha_chars = self.find_abbr_in_word(abbr, tmp_abbr, alpha_chars)
                    if len(tmp_word) > 0:
                        is_find_full_abbr = True
                        # break
            else:
                tmp_word, alpha_chars = self.find_abbr_in_word(abbr, full_abbr, alpha_chars)
                if len(tmp_word) > 0:
                    is_find_full_abbr = True

            if not is_find_full_abbr:
                tmp_word, alpha_chars = self.find_abbr_in_word(abbr, abbr, alpha_chars)

            words = words + tmp_word

        for word in nltk.corpus.stopwords.words('english'):
            if len(word) <= 3:
                continue
            if word in alpha_chars:
                words.append(word)
                alpha_chars = alpha_chars[:alpha_chars.index(word)] + alpha_chars[alpha_chars.index(word) + len(word):]

        if len(alpha_chars) > 0:
            if self.program_D.check(alpha_chars) or self.us_D.check(alpha_chars) or self.gb_D.check(alpha_chars):
                words.append(alpha_chars)
                alpha_chars = ''

        if len(alpha_chars) > 0:
            top = len(alpha_chars)
            bot = 0
            while bot < top:
                for ind in range(bot, top + 1):
                    if ind - bot >= MIN_ABBR_LEN:
                        if alpha_chars[bot:ind] in self.abbreviations:

                            tmp_alpha = alpha_chars[bot:ind]
                            full_abbr = self.abbreviations[tmp_alpha]
                            is_find_fulll_abbr = False
                            if type(full_abbr) == list:
                                for tmp_abbr in full_abbr:
                                    if tmp_abbr in alpha_chars[bot:]:
                                        is_find_fulll_abbr = True
                                        words.append(tmp_abbr)
                                        bot = bot + len(tmp_abbr)

                            else:
                                if full_abbr in alpha_chars[bot:]:
                                    words.append(full_abbr)
                                    is_find_fulll_abbr = True
                                    bot = bot + len(full_abbr)

                            if not is_find_fulll_abbr:
                                words.append(alpha_chars[bot:ind])
                                bot = ind

                            break

                    # min word len for dictionary lookup
                    if ind - bot >= MIN_WORD_LEN:
                        self.enchant_lock.acquire()
                        if self.program_D.check(alpha_chars[bot:ind]):
                            self.enchant_lock.release()
                            tmp_word, tmp_ind = self.find_long_word(alpha_chars, bot, ind + 1, top + 1)
                            if len(tmp_word) > 0:
                                words = words + tmp_word
                                bot = tmp_ind
                            else:
                                words.append(alpha_chars[bot:ind])
                                # tmp_word.append(([alpha_chars[bot:ind]], ind))
                                bot = ind
                            break

                        elif self.us_D.check(alpha_chars[bot:ind]) or self.gb_D.check(alpha_chars[bot:ind]):
                            self.enchant_lock.release()
                            tmp_word, tmp_ind = self.find_long_word(alpha_chars, bot, ind + 1, top + 1)
                            if len(tmp_word) > 0:
                                words = words + tmp_word
                                bot = tmp_ind
                            else:
                                words.append(alpha_chars[bot:ind])
                                # tmp_word.append(([alpha_chars[bot:ind]], ind))
                                bot = ind
                            break
                        self.enchant_lock.release()

                if bot < ind:
                    bot += 1

        return words


    def find_subabbreviations(self, alpha_chars):
        """
            convert strcmp -> [string, compare]
            getlanguagespecificdata -> [get, language, specific, data]
        """

        me = set([])
        if len(alpha_chars) < min( self.config.MIN_MAX_ABBR_LEN, self.config.MIN_MAX_WORD_LEN):
            return me

        if len(alpha_chars) >= self.config.MIN_MAX_ABBR_LEN:
            if alpha_chars in self.abbreviations:

                me.add(alpha_chars)


        if len(alpha_chars) >= self.config.MIN_MAX_WORD_LEN:
            self.enchant_lock.acquire()
            if self.program_D.check(alpha_chars):
                me.add(alpha_chars)
            elif self.us_D.check(alpha_chars) or self.gb_D.check(alpha_chars):
                me.add(alpha_chars)

            self.enchant_lock.release()


        valid_substr_prefix = self.find_subabbreviations(alpha_chars[:-1])
        valid_substr_suffix = self.find_subabbreviations(alpha_chars[1:])

        ### find longest subabbreviations that do not overlap
        #sub_abbrs = valid_substr_suffix.union( valid_substr_prefix )

        return valid_substr_suffix.union( valid_substr_prefix ).union( me )


    def strip_ida_decorations(self, name):
        rules = [ self.ida_import ]
        for rule in rules:
            name = re.sub(rule, "", name)
        return name


    def strip_r2_decorations(self, name):
        """
            Return real name of symbol from r2
        """
        syntax_replace = [
            self.r2_dyn_prefix,
            self.r2_prefix
        ]

        for sf in syntax_replace:
            name = re.sub(sf, "", name)

        return name

    def strip_ida_data_refs(self, name):
        rules = [ self.data_lib ]
        for rule in rules:
            name = re.sub(rule, "", name)
        return name

    def filter_ida_junk(self, iterable):
        return filter(lambda x: not x.startswith("sub_"), iterable)

    def filter_null(self, iterable):
        return filter(lambda x: not x == '', iterable)

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
        return name

    def get_split_subtokens_proc_name(self, s):
        # get sub-tokens. Remove them if len()<1 (single letter or digit)
        return [x for x in [str(x).lower() for x in RE_WORDS.findall(s)] if len(x) > 1]


    def canonical_name(self, name):
        return '_'.join( self.canonical_set(name) )


    def canonical_name_index(self, name):
        canoical_abbrs = self.canonical_set(name)

        abbrs =[]
        for i in canoical_abbrs:
            if i in self.expand_split_abbrs:
                abbrs.extend(self.expand_split_abbrs[i].replace('_', ' ').split())
            else:
                abbrs.append(i)

        tmp_index_dict = {}
        for abbr in abbrs:
            ori_str = name.lower()
            # npos = ori_str.find(abbr)
            all_index = [r.span() for r in re.finditer(abbr, ori_str)]
            tmp_index_dict[abbr] = all_index

        out = find_abbr_index(tmp_index_dict, len(name))

        #out = sorted_label_token(tmp_dict)
        return out


    def canonical_set(self, name):

        words_in_name = name.split()

        labels = set([])

        for word in words_in_name:
            
            #word = word.lower()
            if word.isdigit():
                #labels.add(word)
                continue
            if len(word) >= self.config.MAX_STR_LEN_BEFORE_SEQ_SPLIT:
                #find large words to split the name first
                whole_words = self.quick_subabbreviations(word)
                words = self.subtract_words_sequence(word, whole_words)
            else:
                words = [ word ]

            for subword in words:
                if len(subword) > self.config.MAX_WORD_LEN:
                    labels.add(subword)
                    continue
                
                abbrs = self.find_subabbreviations(subword)


                # print("computing best cut of rod with {} and {}".format(word, abbrs))
                abbr = self.best_cut_of_the_rod(subword, abbrs)

                list_of_lists = list(map(lambda x: re.findall(r'[a-zA-Z]+', x), abbr))

                #add original if no abbreveation or word found
                if len(list_of_lists) == 0:
                    list_of_lists = [ [ subword ] ] 

                new_words = [x for y in list_of_lists for x in y ]

                labels = labels.union( set(new_words) )
        
        return set(filter(lambda x: len(x) >= 2, labels))

    def wordnet_similarity(self, a:str, b:str):
        word_sims = set()
        ac  = self.canonical_set(a)
        bc  = self.canoncial_set(b)
        for aw, bw in itertools.product(ac, bc):
            synset_aw   = wn.synsets(aw)
            synset_bw   = wn.synsets(bw)
            for a_ss, b_ss in itertools.product(synset_aw, synset_bw):
                d = wn.wup_similarity(a_ss, b_ss)
                word_sims.add(d)
        return max(word_sims)


    def alpha_numeric_set(self, name):
        """
            Return set of labels by splitting on all non-alphanumeric letters
        """
        base_name = self.strip_library_decorations(name).lower()

        #remove punctuation
        words_in_name = re.findall(r'[a-zA-Z0-9]+', base_name)
        #numbers_in_name = re.findall(r'[0-9]+', base_name)

        labels = set([])

        for label in words_in_name:
            labels.add(label)

        return labels

    #need to get at least 1/e % of names correct for similarity match
    def check_word_similarity(self, correct_name, inferred_name):
        """
            Check similarity between 2 function names.
            correct name goes first
            :param correct_name: The functions actual name
            :param inferred_name: The name we think it is
            :rtype: Bool
            :return: If similar or not
        """

        if len(correct_name) == 0 and len(inferred_name) == 0:
            return True
        if len(correct_name) == 0 or len(inferred_name) == 0:
            return False

        #correct name is a prefix or suffix of inferref name and vica versa
        if correct_name in inferred_name or inferred_name in correct_name:
            self.logger.debug("Matched on substring! {} -> {}".format(inferred_name, correct_name))
            return True

        #check edit distance as a function of max length
        levehstien_distance = nltk.edit_distance(correct_name, inferred_name )

        #edit distance as a function of length
        m = float( max( len(correct_name),  len(inferred_name) ) )
        #m = float( len(correct_name) + len(inferred_name) )
        #d = float(  abs( len(correct_name) - len(inferred_name) ) )

        #EDIT_DISTANCE_THRESHOLD = 1.0 / 2.0 


        ### higher is better
        edit_sim =  1.0 - ( float(levehstien_distance) / m )
        if edit_sim >= self.config.EDIT_DISTANCE_THRESHOLD:
            self.logger.debug("Matched on edit distance! {} -> {} : {}".format(inferred_name, correct_name, edit_sim))
            return True

        canonical_inferred = self.canonical_name(inferred_name)
        canonical_correct = self.canonical_name(correct_name)

        words_in_inferred_name = re.findall(r'[a-zA-Z]+', canonical_inferred)
        words_in_correct_name = re.findall(r'[a-zA-Z]+', canonical_correct)

        #THRESHOLD = 1.0 / math.e #0.36 -> 1/2, 2/3, 2/4, 2/5, 3/6, ...

        #self.logger.debug("Canonical name: {} => {}".format(inferred_name, words_in_inferred_name))
        #self.logger.debug("Canonical name: {} => {}".format(correct_name, words_in_correct_name))


        ##########################################
        self.enchant_lock.acquire()
        ##########################################
        try:
            #TODO: filter to uniue and WHOLE words! in a dictionary
            words_in_inferred_name = set( words_in_inferred_name )
            words_in_correct_name = set( words_in_correct_name )

            #filter for english words
            words_in_inferred_name = set( filter( lambda w: len(w) > 2 and (self.program_D.check(w) or self.us_D.check(w) or self.gb_D.check(w)), words_in_inferred_name) )
            words_in_correct_name = set( filter( lambda w: len(w) > 2 and (self.program_D.check(w) or self.us_D.check(w) or self.gb_D.check(w)), words_in_correct_name) )

        except Exception as e:
            self.logger.warn("[!] Could not compute check_similarity_of_symbol_name( {} , {} )".format(correct_name, inferred_name))
            self.logger.warn(e)

        finally:
            self.enchant_lock.release()


        #remove boring stop words
        unique_words_inferred = words_in_inferred_name - set(nltk.corpus.stopwords.words('english'))
        unique_words_correct = words_in_correct_name - set(nltk.corpus.stopwords.words('english'))

        stemmer = nltk.stem.PorterStemmer()
        lemmatiser = nltk.stem.wordnet.WordNetLemmatizer()

        stemmed_inferred = set( map( lambda x: stemmer.stem(x), unique_words_inferred) )
        stemmed_correct = set( map( lambda x: stemmer.stem(x), unique_words_correct) )

        lemmatised_inferred = set( map( lambda x: lemmatiser.lemmatize(x), unique_words_inferred) )
        lemmatised_correct = set( map( lambda x: lemmatiser.lemmatize(x), unique_words_correct) )

        if len(lemmatised_correct) > 0 and len(lemmatised_inferred) > 0:
            jaccard_distance = nltk.jaccard_distance( lemmatised_correct, lemmatised_inferred )
            if jaccard_distance < self.config.WORD_MATCH_THRESHOLD:
                self.logger.debug("\tJaccard Distance Lemmatised {} : {} -> {}".format(jaccard_distance, inferred_name, correct_name))
                return True

        if len(stemmed_correct) > 0 and len(stemmed_inferred) > 0:
            jaccard_distance = nltk.jaccard_distance( stemmed_correct, stemmed_inferred )
            if jaccard_distance < 1.0 - self.config.WORD_MATCH_THRESHOLD:
                self.logger.debug("\tJaccard Distance Stemmed {} : {} -> {}".format(jaccard_distance, inferred_name, correct_name))
                return True

        if len(unique_words_correct) > 0 and len(unique_words_inferred) > 0:
            if self.compare_synsets(unique_words_correct, unique_words_inferred) >= 0.385:
                self.logger.debug("\tMatched on wordnet synsets: {} -> {}".format(inferred_name, correct_name))
                return True

        return False

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
                    self.logger.debug("Matched ({},{}) on {}".format(_A[i], _B[j], a_synonyms.intersection(b_synonyms)))
                    WORD_MATCH = True
                    break

            if WORD_MATCH:
                mc += 1.0

        return mc / max_m


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

    def distance(self, a, b):
        """
            Get the distance between a and b, smaller is closer distance
        """
        d = self.score(a, b)
        return 1.0 / (1.0 + np.log(1+d))
