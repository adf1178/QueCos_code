import os
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging
from tqdm import tqdm
import math
from data_loader import *
from mydata import mydataset_test
import torch
from utils import normalize, similarity, sent2indexes
import models
import configs
batchsize = 8
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
config = getattr(configs, 'config_JointEmbeder')()
model = getattr(models, 'JointEmbeder')(config)  # initialize the model
ckpt = f"/data/home/zhnong/deepcs/DCS-master/pytorch/output/JointEmbeder/CSN-Java/models/step1400000.h5"
# ckpt = f'./output/JointEmbeder/github/models/epo630000.h5'


model.load_state_dict(torch.load(ckpt, map_location=device))
data_path = "./Code2Seq/"

# testset = eval(config['dataset_name'])(data_path,
#                                   "test.name.h5", config['name_len'],
#                                   "test.apiseq.h5", config['api_len'],
#                                   "test.unordered.tokens.h5", config['tokens_len'],
#                                   "test.desc.h5", config['desc_len'])
testset = mydataset_test(6, 50, 30, 30)
k = 0


def validate(valid_set, model, pool_size, K, sim_measure):
    """
    simple validation in a code pool.
    @param: poolsize - size of the code pool, if -1, load the whole test set
    """
    def ACC(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum+1
        return sum/float(len(real))

    def MAP(real, predict):
        sum = 0.0
        for id, val in enumerate(real):
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum+(id+1)/float(index+1)
        return sum/float(len(real))

    def MRR(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1:
                sum = sum+1.0/float(index+1)
        return sum/float(len(real))

    def NDCG(real, predict):
        dcg = 0.0
        idcg = IDCG(len(real))
        for i, predictItem in enumerate(predict):
            if predictItem in real:
                itemRelevance = 1
                rank = i+1
                dcg += (math.pow(2, itemRelevance)-1.0) * \
                    (math.log(2)/math.log(rank+1))
        return dcg/float(idcg)

    def IDCG(n):
        idcg = 0
        itemRelevance = 1
        for i in range(n):
            idcg += (math.pow(2, itemRelevance)-1.0) * \
                (math.log(2)/math.log(i+2))
        return idcg

    model.eval()
    device = next(model.parameters()).device

    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=1000,
                                              shuffle=True, drop_last=True, num_workers=1)
    r1s, r5s, r10s, mrrs = [], [], [], []
    code_reprs, desc_reprs = [], []
    n_processed = 0
    for batch in tqdm(data_loader):
        if len(batch) == 8:  # names, name_len, toks, index, descs, desc_len, bad_descs, bad_desc_len
            code_batch = [tensor.to(device) for tensor in batch[:4]]
            desc_batch = [tensor.to(device) for tensor in batch[4:6]]
        with torch.no_grad():
            code_repr = model.code_encoding(
                *code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(
                *desc_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
            if sim_measure == 'cos':
                code_repr = normalize(code_repr)
                desc_repr = normalize(desc_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

    for k in tqdm(range(0, n_processed, pool_size)):
        code_pool, desc_pool = code_reprs[k:k +
                                          pool_size], desc_reprs[k:k+pool_size]
        if pool_size == len(desc_pool):
            # for i in range(pool_size):
            for i in range(min(10000, pool_size)):
                desc_vec = np.expand_dims(desc_pool[i], axis=0)  # [1 x dim]
                n_results = K
                if sim_measure == 'cos':
                    sims = np.dot(code_pool, desc_vec.T)[:, 0]  # [pool_size]
                else:
                    sims = similarity(code_pool, desc_vec,
                                      sim_measure)  # [pool_size]

                negsims = np.negative(sims)
                predict = np.argsort(negsims)
                predict1 = predict[:1]
                predict5 = predict[:5]
                predict10 = predict[:10]
                predict1 = [int(k) for k in predict1]
                predict5 = [int(k) for k in predict5]
                predict10 = [int(k) for k in predict10]
                predict = [int(k) for k in predict]
                real = [i]
                r1s.append(ACC(real, predict1))
                r5s.append(ACC(real, predict5))
                r10s.append(ACC(real, predict10))
                mrrs.append(MRR(real, predict))
    return {'r1': np.mean(r1s), 'r5': np.mean(r5s), 'r10': np.mean(r10s), 'mrr': np.mean(mrrs)}


# this is the test function, you can modify pool size and VALID or TEST set
testresult = validate(testset, model, 1000, 1, config['sim_measure'])
print(testresult)

print("done")
