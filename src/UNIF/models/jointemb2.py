from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as weight_init
from torch import optim
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)
parentPath = os.path.abspath("..")
sys.path.insert(0, parentPath)# add parent folder to path so as to import common modules
from modules import SeqEncoder, BOWEncoder, Transformer
class Config(object):
    N = 1 #6 in Transformer Paper
    d_model = 256 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    h = 8
    dropout = 0.1
    output_size = 128
    lr = 0.01
    max_epochs = 100
    # batch_size = 64
    # max_sen_len = 60

class JointEmbeder(nn.Module):
    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.conf = config
        self.margin = config['margin']
        self.config1 = Config()  
        # self.name_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        # self.api_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        # self.tok_encoder=BOWEncoder(config['n_words'],config['emb_size'],config['n_hidden'])
        # self.desc_encoder=SeqEncoder(config['n_words'],config['emb_size'],config['lstm_dims'])
        self.tok_encoder=Transformer(self.config1,config['n_words'])
        self.desc_encoder=Transformer(self.config1,config['n_words'])
        #self.fuse1=nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden'])
        #self.fuse2 = nn.Sequential(
        #    nn.Linear(config['emb_size']+4*config['lstm_dims'], config['n_hidden']),
        #    nn.BatchNorm1d(config['n_hidden'], eps=1e-05, momentum=0.1),
        #    nn.ReLU(),
        #    nn.Linear(config['n_hidden'], config['n_hidden']),
        #)

        # self.w_name = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        # self.w_api = nn.Linear(2*config['lstm_dims'], config['n_hidden'])
        # self.w_tok = nn.Linear(config['emb_size'], config['n_hidden'])
        # self.fuse3 = nn.Linear(config['n_hidden'], config['n_hidden'])
        
        # self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])        
        # self.extend = nn.Linear(100,1000)
        # self.init_weights()
        
    def init_weights(self):# Initialize Linear Weight 
        for m in [self.w_name, self.w_api, self.w_tok, self.fuse3]:        
            m.weight.data.uniform_(-0.05, 0.05)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.) 
            
    def code_encoding(self, name, name_len, api, api_len, tokens, tok_len):
        # name_repr=self.name_encoder(name, name_len)
        # api_repr=self.api_encoder(api, api_len)
        # tok_repr=self.tok_encoder(tokens, tok_len)
        tok_repr=self.tok_encoder(tokens)

        #code_repr= self.fuse2(torch.cat((name_repr, api_repr, tok_repr),1))
        # code_repr = self.fuse3(torch.tanh(self.w_name(name_repr)+self.w_api(api_repr)+self.w_tok(tok_repr)))
        return tok_repr
        
    def desc_encoding(self, desc, desc_len):
        # desc_repr=self.desc_encoder(desc, desc_len)
        desc_repr=self.desc_encoder(desc)
        return desc_repr
    
    def forward(self, name, name_len, apiseq, api_len, tokens, tok_len, desc_good, desc_good_len, desc_bad, desc_bad_len):
   
        batch_size=name.size(0)
        code_repr=self.code_encoding(name, name_len, apiseq, api_len, tokens, tok_len)
        # code_repr = self.extend(code_repr)  # 100->1000

        desc_good_repr=self.desc_encoding(desc_good, desc_good_len)
        desc_bad_repr=self.desc_encoding(desc_bad, desc_bad_len)

        # print(desc_bad_repr.shape,"        ",code_repr.shape)
        good_sim=F.cosine_similarity(code_repr, desc_good_repr)
        bad_sim=F.cosine_similarity(code_repr, desc_bad_repr) # [batch_sz x 1]
        
        loss=(self.margin-good_sim+bad_sim).clamp(min=1e-6).mean()
        
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        
        return loss