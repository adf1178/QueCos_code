import os
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class AttCodeEncoder(nn.Module):
    '''
    https://medium.com/data-from-the-trenches/how-deep-does-your-sentence-embedding-model-need-to-be-cdffa191cb53
    https://www.kdnuggets.com/2019/10/beyond-word-embedding-document-embedding.html
    https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d#bbe8
    '''
    def __init__(self, vocab_size, emb_size, tokens_len, hidden_size):
        super(AttCodeEncoder, self).__init__()
        self.emb_size=emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        #self.word_weights = get_word_weights(vocab_size)
        self.attention = nn.Linear(emb_size,1)
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.attention.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)
        
    def forward(self, input, input_len=None): 
        batch_size, seq_len =input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded= F.dropout(embedded, 0.25, self.training) # [batch_size x seq_len x emb_size]
        #print(input.size())
        #model = fasttext.train_unsupervised('./data/github/train.tokens.h5', model="skipgram", lr=0.05, dim=100) 
        #           ws=5, epoch=5, minCount=5, 
        #           minCountLabel=0, minn=3, 
        #           maxn=6, neg=5, wordNgrams=1, 
        #           loss="ns", bucket=2000000, 
        #           thread=12, lrUpdateRate=100,
        #           t=1e-4, label="__label__", 
        #           verbose=2, pretrainedVectors="")
        
        
        # try to use a weighting scheme to summarize bag of word embeddings: 
        # for example, a smooth inverse frequency weighting algorithm: https://github.com/peter3125/sentence2vec/blob/master/sentence2vec.py
        # word_weights = self.word_weights(input) # [batch_size x seq_len x 1]
        # embeded = word_weights*embedded 
        inital_value = torch.exp(self.attention(embedded).squeeze(2))
        test = torch.sum(inital_value, 1, True)
        attention_weight = torch.div(inital_value, torch.sum(inital_value, 1, True))
        attention_weight = attention_weight.unsqueeze(-1)
        output = torch.bmm(embedded.transpose(1,2), attention_weight).squeeze(2)
       
        return output

class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers=1):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.init_weights()
        
    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)

    def forward(self, inputs, input_lens=None): 
        batch_size, seq_len=inputs.size()
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = F.dropout(inputs, 0.25, self.training)
        encoding = torch.sum(inputs, 1, True) / seq_len
        encoding = encoding.squeeze(1)


        return encoding #pooled_encoding

    
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)    
    

def get_word_weights(vocab_size, padding_idx=0):
    '''contruct a word weighting table '''
    def cal_weight(word_idx):
        return 1-math.exp(-word_idx)
    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:        
        weight_table[padding_idx] = 0. # zero vector for padding dimension
    return torch.FloatTensor(weight_table)

 
 
 
 