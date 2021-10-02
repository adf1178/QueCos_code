
def config_JointEmbeder():   
    conf = {
        # data_params
        'dataset_name':'CodeSearchDataset', # name of dataset to specify a data loader
            #training data
            'train_tokens':'train.unordered.tokens.h5',
            'train_desc':'train.desc.h5',
            #test data
            'valid_tokens':'valid.unordered.tokens.h5',
            'valid_desc':'valid.desc.h5',
            #use data (computing code vectors)
            'use_codebase':'use.rawcode.txt',#'use.rawcode.h5'
            'use_tokens':'use.tokens.h5',     
            #results data(code vectors)            
            'use_codevecs':'use.codevecs.h5',  
            'train_name':'train.name.h5',
            'train_api':'train.unordered.apiseq.h5',
            'train_tokens':'train.unordered.tokens.h5',
            'train_desc':'train.desc.h5',
            #test data
            'valid_name':'valid.name.h5',
            'valid_api':'valid.apiseq.h5',
            'valid_tokens':'valid.unordered.tokens.h5',
            'valid_desc':'valid.desc.h5',
            #use data (computing code vectors)
            'use_codebase':'use.rawcode.txt',#'use.rawcode.h5'
            'use_names':'use.name.h5',
            'use_apis':'use.apiseq.h5',
            'use_tokens':'use.tokens.h5',     
            #results data(code vectors)            
            'use_codevecs':'use.codevecs.normalized.h5',#'use.codevecs.h5',               
                   
            #parameters
            'name_len': 6,
            'api_len':30,
            'tokens_len':50,
            'desc_len': 50,
            'n_words': 10000, # len(vocabulary) + 1
            #vocabulary info
            'vocab_tokens':'vocab.tokens.json',
            'vocab_desc':'vocab.desc.json',
                    
        #training_params            
            'batch_size': 64,
            'chunk_size':200000,
            'nb_epoch': 100,
            #'optimizer': 'adam',
            'learning_rate':1e-3,
            'adam_epsilon':1e-8,
            'warmup_steps':5000,
            'fp16': False,
            'fp16_opt_level': 'O1', #For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].
                            #"See details at https://nvidia.github.io/apex/amp.html"

        # model_params
            'emb_size': 512,
            'n_hidden': 1024,#number of hidden dimension of code/desc representation
            # recurrent
            'lstm_dims': 512, # * 2          
            'margin': 0.6,
            'sim_measure':'cos',#similarity measure: cos, poly, sigmoid, euc, gesd, aesd. see https://arxiv.org/pdf/1508.01585.pdf
                         #cos, poly and sigmoid are fast with simple dot, while euc, gesd and aesd are slow with vector normalization.
    }
    return conf

