from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch
import torch.nn as nn
import os
import numpy as np
from typing import *
import math
from tqdm import tqdm
import os
import json
import jsonlines

def data(t = 'train'):
    with open(os.path.join(os.path.dirname(
            __file__), "../../../../../data-java/{}.jsonl".format(t)), "r+", encoding="utf8") as f:
        return [item for item in jsonlines.Reader(f)]


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained(os.path.join(os.path.dirname(__file__),  "python_model")).eval().cuda()

def eval(queries: List[str], codes: List[str], pool_size=1000, code_cache_vec: List[torch.Tensor] = [], verbose=False):
    assert(len(queries) <= len(codes))
    assert(len(codes) + len(code_cache_vec) == pool_size)
    length = len(queries)
    assert(length <= pool_size)
    queries_vec = [model(tokenizer(i,return_tensors='pt')['input_ids'][:,:512].cuda())[1].detach() for i in (tqdm(queries) if verbose else queries)]
    codes_vec = [model(tokenizer(i,return_tensors='pt')['input_ids'][:,:512].cuda())[1].detach() for i in (tqdm(codes) if verbose else codes)] + code_cache_vec
    queries_vec = torch.cat(queries_vec, 0)
    codes_vec = torch.cat(codes_vec, 0)
    scores=torch.einsum("ab,cb->ac",queries_vec,codes_vec)
    scores=torch.softmax(scores,-1)

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

    r1s, r5s, r10s, mrrs = [], [], [], []
    for i, score in enumerate(scores.detach().cpu().numpy()):
        if i >= length:
            break
        predict = np.argsort(-score)
        real = [i]
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

def eval_integrate(queries1: List[str], queries2: List[str], codes: List[str], pool_size=1000, code_cache_vec: List[torch.Tensor] = [], beta=0.5,verbose=False):
    assert(len(queries1) == len(queries2))
    assert(len(queries1) <= len(codes))
    assert(len(codes) + len(code_cache_vec) == pool_size)
    length = len(queries1)
    assert(length <= pool_size)
    queries1_vec = [model(tokenizer(i,return_tensors='pt')['input_ids'][:,:512].cuda())[1].detach() for i in (tqdm(queries1) if verbose else queries1)]
    queries2_vec = [model(tokenizer(i,return_tensors='pt')['input_ids'][:,:512].cuda())[1].detach() for i in (tqdm(queries2) if verbose else queries2)]
    codes_vec = [model(tokenizer(i,return_tensors='pt')['input_ids'][:,:512].cuda())[1].detach() for i in (tqdm(codes) if verbose else codes)] + code_cache_vec
    queries1_vec = torch.cat(queries1_vec, 0)
    queries2_vec = torch.cat(queries2_vec, 0)
    codes_vec = torch.cat(codes_vec, 0)
    scores1=torch.einsum("ab,cb->ac",queries1_vec,codes_vec)
    scores1=torch.softmax(scores1,-1)
    scores2=torch.einsum("ab,cb->ac",queries2_vec,codes_vec)
    scores2=torch.softmax(scores2,-1)
    scores = beta*scores1 + (1-beta)*scores2

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

    r1s, r5s, r10s, mrrs = [], [], [], []
    for i, score in enumerate(scores.detach().cpu().numpy()):
        if i >= length:
            break
        predict = np.argsort(-score)
        real = [i]
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

pad_data = data('valid')
pad_data_vec_cache: Dict[int, torch.Tensor] = {}
def eval_easy_to_use(queries: List[str], codes: List[str], pool_size = 1000, verbose=False):
    length = len(queries)
    r1s, r5s, r10s, mrrs = [], [], [], []
    for i in range(math.ceil(length / pool_size)):
        queries_batch =  queries[i * pool_size: (i+1)*pool_size]
        codes_batch =  codes[i * pool_size: (i+1)*pool_size]
        code_cache_vec = []
        while len(codes_batch) + len(code_cache_vec) < pool_size:
            rand_index = np.random.randint(0, len(pad_data))
            if rand_index in pad_data_vec_cache:
                pad_code_vec = pad_data_vec_cache[rand_index]
            else:
                pad_code = pad_data[rand_index]['code']
                pad_code_vec = model(tokenizer(pad_code,return_tensors='pt')['input_ids'][:,:512].cuda())[1].detach()
                pad_data_vec_cache[rand_index] = pad_code_vec
            code_cache_vec += [pad_code_vec]
        result = eval(queries_batch, codes_batch, pool_size, code_cache_vec, verbose)
        r1s += [result['r1']]
        r5s += [result['r5']]
        r10s += [result['r10']]
        mrrs += result['mrrs']
    return {'r1': np.mean(r1s), 'r5': np.mean(r5s), 'r10': np.mean(r10s), 'mrr': np.mean(mrrs), 'mrrs': mrrs}

def eval_integrate_easy_to_use(queries1: List[str], queries2: List[str], codes: List[str], pool_size = 1000, beta=0.5, verbose=False):
    length = len(queries1)
    r1s, r5s, r10s, mrrs = [], [], [], []
    for i in range(math.ceil(length / pool_size)):
        queries1_batch =  queries1[i * pool_size: (i+1)*pool_size]
        queries2_batch =  queries2[i * pool_size: (i+1)*pool_size]
        codes_batch =  codes[i * pool_size: (i+1)*pool_size]
        code_cache_vec = []
        while len(codes_batch) + len(code_cache_vec) < pool_size:
            rand_index = np.random.randint(0, len(pad_data))
            if rand_index in pad_data_vec_cache:
                pad_code_vec = pad_data_vec_cache[rand_index]
            else:
                pad_code = pad_data[rand_index]['code']
                pad_code_vec = model(tokenizer(pad_code,return_tensors='pt')['input_ids'][:,:512].cuda())[1].detach()
                pad_data_vec_cache[rand_index] = pad_code_vec
            code_cache_vec += [pad_code_vec]
        result = eval_integrate(queries1_batch, queries2_batch, codes_batch, pool_size, code_cache_vec, beta=beta, verbose=verbose)
        r1s += [result['r1']]
        r5s += [result['r5']]
        r10s += [result['r10']]
        mrrs += result['mrrs']
    return {'r1': np.mean(r1s), 'r5': np.mean(r5s), 'r10': np.mean(r10s), 'mrr': np.mean(mrrs), 'mrrs': mrrs}



# query = "set a variable as hello world"
# query_vec = model(tokenizer(query,return_tensors='pt')['input_ids'])[1]
# code_1="print('hello world')"
# code1_vec = model(tokenizer(code_1,return_tensors='pt')['input_ids'])[1]
# code_2="s = 'hello world'"
# code2_vec = model(tokenizer(code_2,return_tensors='pt')['input_ids'])[1]
# code_3="hello world"
# code3_vec = model(tokenizer(code_3,return_tensors='pt')['input_ids'])[1]
# code_vecs=torch.cat((code1_vec,code2_vec,code3_vec),0)
# codes = [code_1,code_2,code_3]
# scores=torch.einsum("ab,cb->ac",query_vec,code_vecs)
# scores=torch.softmax(scores,-1)
# print("Query:",query)
# for i in range(3):
#     print("Code:",codes[i])
#     print("Score:",scores[0,i].item())



# query = "Download an image and save the content in output_dir"
# query_vec = model(tokenizer(query,return_tensors='pt')['input_ids'])[1]
# code_1="""
# def f(image_url, output_dir):
#     import requests
#     r = requests.get(image_url)
#     with open(output_dir, 'wb') as f:
#         f.write(r.content)
# """
# code1_vec = model(tokenizer(code_1,return_tensors='pt')['input_ids'])[1]
# code_2="""
# def f(image, output_dir):
#     with open(output_dir, 'wb') as f:
#         f.write(image)
# """
# code2_vec = model(tokenizer(code_2,return_tensors='pt')['input_ids'])[1]
# code_3="""
# def f(image_url, output_dir):
#     import requests
#     r = requests.get(image_url)
#     return r.content
# """
# code3_vec = model(tokenizer(code_3,return_tensors='pt')['input_ids'])[1]
# code_vecs=torch.cat((code1_vec,code2_vec,code3_vec),0)
# codes = [code_1,code_2,code_3]
# scores=torch.einsum("ab,cb->ac",query_vec,code_vecs)
# scores=torch.softmax(scores,-1)
# print("")
# print("Query:",query)
# for i in range(3):
#     print("Code:",codes[i])
#     print("Score:",scores[0,i].item())

