import os
import sys
import traceback
import numpy as np
import argparse
import threading
import codecs
import logging

from typing import *
from torch import Tensor
from tqdm import tqdm
import math
from .data_loader import *

import torch
from .mydata import mydataset_test, mydataset, mydataset_valid
from .utils import normalize, similarity, sent2indexes
from .data_loader import load_dict, load_vecs
from . import models, configs

batchsize = 8
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
config = getattr(configs, 'config_JointEmbeder')()
model = getattr(models, 'JointEmbeder')(config)  # initialize the model
ckpt = f'/data/czwang/ke/src/deepcs/output/JointEmbeder/CSN-Java/models/step1400000.h5'
# ckpt = f'/data/czwang/ke/src/deepcs/output/JointEmbeder/CSN-python/models/step1270000.h5'
# ckpt = f'./output/JointEmbeder/github/models/epo630000.h5'


model.load_state_dict(torch.load(ckpt, map_location=device))
# data_path = "./unif-python-data/"
# testset = eval(config['dataset_name'])(data_path,
#                                        "test.name.h5", 6,
#                                        "test.desc.h5", config['desc_len'])
data_path = "./Code2Seq/"
valid_set = mydataset_valid(6, 50, 30, 30)
testset = mydataset_test(6, 50, 30, 30)
valid_seg = [valid_set[i][:6] + valid_set[i][4:6] for i in range(1000)]
k = 0


def validate(valid_set, model, pool_size, K, sim_measure, ke=None, alpha=1):
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
    BOS=2
    model.eval()
    device = next(model.parameters()).device

    batch_size = 1000
    data_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size,
                                              shuffle=True, drop_last=True, num_workers=1)
    accs, mrrs, maps, ndcgs = [], [], [], []
    code_reprs, desc_reprs, query_reprs = [], [], []
    r1s, r5s, r10s = [], [], []
    n_processed = 0
    for batch in tqdm(data_loader):
        code_batch = [tensor.to(device) for tensor in batch[:4]]
        desc_batch = [tensor.to(device) for tensor in batch[4:6]]

        if ke is not None:
            # 使用ke对query进行处理
            queryVec = desc_batch[0].transpose(
                0, 1)  # [query_len x batch_size]
            maxQueryLen = queryVec.max(axis=1).values.ne(
                0).sum().item()  # 当前batch中所有query的最大长度
            attention_mask = torch.LongTensor(queryVec).data.eq(
                0).t()[:, :maxQueryLen]  # [batch_size x maxQueryLen]
            attention_mask = attention_mask.cuda()
            if hasattr(ke, 'decoder') and hasattr(ke.decoder, 'attn'):
                ke.decoder.attn.applyMask(attention_mask)
            count = len(queryVec)
            # descGenVec = ke.translate([(torch.LongTensor(queryVec).cuda(), desc_batch[1].cuda()), None, 
            #     torch.LongTensor([[BOS]*count]).cuda(), None, None, None], 120)  # [max_desc_len x batch_size]
            descVec, _ = ke.sample([(queryVec.cuda(), desc_batch[1].cuda()), None, torch.LongTensor(
                [[0]*1000]).cuda(), None, None, None], 120)  # [max_desc_len x batch_size]
            desc_batch = (descVec.transpose(0, 1).cpu(), torch.LongTensor([
                          len(i) for i in descVec.transpose(0, 1).cpu()]))

        with torch.no_grad():
            code_repr = model.code_encoding(
                *code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(
                *desc_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
            if sim_measure == 'cos' or sim_measure == 'cos_integrate':
                code_repr = normalize(code_repr)
                desc_repr = normalize(desc_repr)
            if sim_measure == 'cos_integrate':
                qury_batch = [tensor.to(device) for tensor in batch[4:6]]
                query_repr = model.desc_encoding(
                    *qury_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
                query_repr = normalize(query_repr)
                query_reprs.append(query_repr)

        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        n_processed += batch[0].size(0)
    if sim_measure == 'cos_integrate':
        code_reprs, desc_reprs, query_reprs = np.vstack(
            code_reprs), np.vstack(desc_reprs), np.vstack(query_reprs)
    else:
        code_reprs, desc_reprs = np.vstack(code_reprs), np.vstack(desc_reprs)

    for k in tqdm(range(0, n_processed, pool_size)):
        code_pool, desc_pool = code_reprs[k:k +
                                          pool_size], desc_reprs[k:k+pool_size]
        for i in range(min(10000, pool_size)):  # for i in range(pool_size):
            desc_vec = np.expand_dims(desc_pool[i], axis=0)  # [1 x dim]
            n_results = K
            if sim_measure == 'cos':
                sims = np.dot(code_pool, desc_vec.T)[:, 0]  # [pool_size]
            elif sim_measure == 'cos_integrate':
                assert ke is not None
                query_pool = query_reprs[k:k+pool_size]
                query_vec = np.expand_dims(query_pool[i], axis=0)  # [1 x dim]
                sims1 = np.dot(code_pool, desc_vec.T)[:, 0]
                sims2 = np.dot(code_pool, query_vec.T)[:, 0]
                sims = alpha*sims1 + (1-alpha)*sims2
            else:
                sims = similarity(code_pool, desc_vec,
                                  sim_measure)  # [pool_size]

            negsims = np.negative(sims)
            # predict = np.argpartition(negsims, kth=n_results-1)#predict=np.argsort(negsims)#
            # predict = predict[:n_results]
            # predict = [int(k) for k in predict]
            # real = [i]
            # accs.append(ACC(real,predict))
            # mrrs.append(MRR(real,predict))
            # maps.append(MAP(real,predict))
            # ndcgs.append(NDCG(real,predict))
            predict = np.argsort(negsims)
            predict1 = predict[:1]
            predict5 = predict[:5]
            predict10 = predict[:10]

            predict = [int(k) for k in predict]
            predict1 = [int(k) for k in predict1]
            predict5 = [int(k) for k in predict5]
            predict10 = [int(k) for k in predict10]
            real = [i]
            r1s.append(ACC(real, predict1))
            r5s.append(ACC(real, predict5))
            r10s.append(ACC(real, predict10))
            mrrs.append(MRR(real, predict))
    return {'r1': np.mean(r1s), 'r5': np.mean(r5s), 'r10': np.mean(r10s), 'mrr': np.mean(mrrs)}


valid_seg_vec_cache = None


def validate_for_segments(
    valid_segments: List[
        Tuple[
            np.array,  # name
            int,  # name_len
            np.array,   # codeVec
            int,  # code_len
            np.array,  # desc
            int,  # desc_len
        ]
    ],
    model,
    pool_size,
    K,
    sim_measure,
    alpha=1.0
):
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

    global valid_seg_vec_cache
    if valid_seg_vec_cache is None:
        valid_seg_vec_cache = ([], [])
        data_loader = torch.utils.data.DataLoader(dataset=valid_seg, batch_size=50,
                                                  shuffle=False, drop_last=True, num_workers=1)
        for batch in tqdm(data_loader):
            code_batch = [tensor.to(device) for tensor in batch[:4]]
            desc_batch = [tensor.to(device) for tensor in batch[4:6]]
            with torch.no_grad():
                code_repr = model.code_encoding(
                    *code_batch).data.cpu().numpy().astype(np.float32)
                desc_repr = model.desc_encoding(
                    *desc_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
                if sim_measure == 'cos' or sim_measure == 'cos_integrate':
                    code_repr = normalize(code_repr)
                    desc_repr = normalize(desc_repr)
                valid_seg_vec_cache[0].append(code_repr)
                valid_seg_vec_cache[1].append(desc_repr)

    accs, mrrs, maps, ndcgs = [], [], [], []
    code_reprs, desc_reprs, query_reprs = [], [], []
    r1s, r5s, r10s = [], [], []
    n_processed = 0
    _valid_segments = []
    for valid_segment in valid_segments:
        if sim_measure == 'cos_integrate':
            _valid_segment = [None, None, None, None, None, None, None, None]
        else:
            _valid_segment = [None, None, None, None, None, None]
        # name
        _valid_segment[0] = valid_set.pad_seq(valid_segment[0], 6)
        _valid_segment[1] = min(6, valid_segment[1])
        # code
        _valid_segment[2] = valid_set.pad_seq(valid_segment[2], 50)
        _valid_segment[3] = min(50, valid_segment[3])
        # desc
        _valid_segment[4] = valid_set.pad_seq(valid_segment[4], 30)
        _valid_segment[5] = min(30, valid_segment[5])
        if sim_measure == 'cos_integrate':
            _valid_segment[6] = valid_set.pad_seq(valid_segment[6], 30)
            _valid_segment[7] = min(30, valid_segment[7])

        _valid_segments.append(_valid_segment)
    data_loader = torch.utils.data.DataLoader(dataset=(_valid_segments + valid_seg), batch_size=50,
                                              shuffle=False, drop_last=False, num_workers=1)
    for batch in data_loader:
        code_batch = [torch.LongTensor(tensor).to(device)
                      for tensor in batch[:4]]
        desc_batch = [torch.LongTensor(tensor).to(device)
                      for tensor in batch[4:6]]
        if sim_measure == 'cos_integrate':
            query_batch = [torch.LongTensor(tensor).to(device)
                           for tensor in batch[6:8]]
        else:
            query_batch = []
        with torch.no_grad():
            code_repr = model.code_encoding(
                *code_batch).data.cpu().numpy().astype(np.float32)
            desc_repr = model.desc_encoding(
                *desc_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
            if sim_measure == 'cos':
                code_repr = normalize(code_repr)
                desc_repr = normalize(desc_repr)
            if sim_measure == 'cos_integrate':
                query_repr = model.desc_encoding(
                    *query_batch).data.cpu().numpy().astype(np.float32)  # [poolsize x hid_size]
                code_repr = normalize(code_repr)
                desc_repr = normalize(desc_repr)
                query_repr = normalize(query_repr)
        code_reprs.append(code_repr)
        desc_reprs.append(desc_repr)
        if sim_measure == 'cos_integrate':
            query_reprs.append(query_repr)
        n_processed += batch[0].size(0)
    code_reprs, desc_reprs = np.vstack(code_reprs + valid_seg_vec_cache[0])[
        :pool_size], np.vstack(desc_reprs + valid_seg_vec_cache[1])[:pool_size]
    if sim_measure == 'cos_integrate':
        query_reprs = np.vstack(
            query_reprs + valid_seg_vec_cache[1])[:pool_size]

    for k in range(0, len(valid_segments), pool_size):
        code_pool, desc_pool = code_reprs[k:k +
                                          pool_size], desc_reprs[k:k+pool_size]
        for i in range(pool_size):
            if i + k >= len(valid_segments):
                break
            desc_vec = np.expand_dims(desc_pool[i], axis=0)  # [1 x dim]
            n_results = K
            if sim_measure == 'cos':
                sims = np.dot(code_pool, desc_vec.T)[:, 0]  # [pool_size]
            elif sim_measure == 'cos_integrate':
                query_pool = query_reprs[k:k+pool_size]
                query_vec = np.expand_dims(query_pool[i], axis=0)  # [1 x dim]
                sims1 = np.dot(code_pool, desc_vec.T)[:, 0]
                sims2 = np.dot(code_pool, query_vec.T)[:, 0]
                sims = alpha*sims1 + (1-alpha)*sims2
            else:
                sims = similarity(code_pool, desc_vec,
                                  sim_measure)  # [pool_size]

            negsims = np.negative(sims)
            # predict = np.argpartition(negsims, kth=n_results-1)#predict=np.argsort(negsims)#
            # predict = predict[:n_results]
            # predict = [int(k) for k in predict]
            # real = [i]
            # accs.append(ACC(real,predict))
            # mrrs.append(MRR(real,predict))
            # maps.append(MAP(real,predict))
            # ndcgs.append(NDCG(real,predict))
            predict = np.argsort(negsims)
            predict1 = predict[:1]
            predict5 = predict[:5]
            predict10 = predict[:10]

            predict = [int(k) for k in predict]
            predict1 = [int(k) for k in predict1]
            predict5 = [int(k) for k in predict5]
            predict10 = [int(k) for k in predict10]
            real = [i]
            r1s.append(ACC(real, predict1))
            r5s.append(ACC(real, predict5))
            r10s.append(ACC(real, predict10))
            mrrs.append(MRR(real, predict))
    return {'r1': np.mean(r1s), 'r5': np.mean(r5s), 'r10': np.mean(r10s), 'mrr': np.mean(mrrs), 'mrrs': mrrs}
