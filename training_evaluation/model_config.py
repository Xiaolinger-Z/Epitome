
class Config:
    device = '0'
    multi_gpu = True
    num_workers = 4

    # #'location of the data corpus and test dataset'
    datapre = ''
    test_datapre =''
    #assembly instruction vocab path which is pretrain model generate
    node_vocab_path  =  ''
    #'function name vocab path'
    graph_label_vocab_path = ''
    # pre-train model path
    pre_train_model ='../pre_train/pre_train_model/modelout/best_ceckpoint'

    word_net_path = 'X64_wordnet.json'
    save_path = './modelout/'
    test_opt = ['O0', 'O1', 'O2', 'O3', 'Os']


    lr = 1e-4  #'Initial learning rate'
    min_lr = 1e-6 # #'Minimum learning rate.'
    dropout = 0.1 # #'Dropout rate (1 - keep probability)
    target_len = 10 # #'function name length'
    node_len = 16  #'instruction length'
    node_num = 100  #'the number of node in fined-grained CFG'
    num_blocks = 6 # the num block of transformer
    num_heads = 8 # the num head of transformer
    batch_size = 64  #'Input batch size for training'
    epochs = 200 #'Number of epochs to train'
    emb_dim = 128  #'Size of embedding'
    conv_feature_dim = 128 #'Size of conv layer in node embedding'
    hidden_dim = 256 #'Size of hidden size'
    feature_num = 320  # the size of select feature num
    radius = 2  # Diameter of ego networks'
    sinusoid = False
    beam = 3 # 'beam size'
    accumulate_step = 1

    # 'Factor in the ReduceLROnPlateau learning rate scheduler'
    factor = 0.5
    #Patience in the ReduceLROnPlateau learning rate scheduler
    patience = 3
    #random seed
    seed = 42

    # config for function name split and normalize NLP
    MAX_STR_LEN_BEFORE_SEQ_SPLIT = 10
    MIN_MAX_WORD_LEN = 3
    MIN_MAX_ABBR_LEN = 2
    EDIT_DISTANCE_THRESHOLD = 0.66
    WORD_MATCH_THRESHOLD = 0.36787968862663154641
    MAX_WORD_LEN = 13
