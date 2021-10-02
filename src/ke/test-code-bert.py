import os
import numpy as np
import torch
import json
from ..CodeBERT.CodeBERT.codesearch.demo.demo import eval_integrate_easy_to_use

poolsize = 1000
batch_size = 1000
beta=1.0
testset = np.load(os.path.join(os.path.dirname(__file__),
                               "../../data-python4csn/test.npy"), allow_pickle=True).item()
ke_path = os.path.join(os.path.dirname(
    __file__), '../../save/model_xentke_python_mrr_code_bert_transformer/model_xentke_python_mrr_code_bert_transformer_19.pt')
with open(os.path.join(os.path.dirname(__file__), '../../data-python4csn/descri.json')) as f:
    labelToIdx = json.loads(f.read())
    discri_dict = {labelToIdx[i]: i for i in labelToIdx}
#######################
load = torch.load(ke_path)
ke = load['model']

BOS = 1


def pad_seq(seq, maxlen):
    if len(seq) < maxlen:
        # !!!!! numpy appending is slow. Try to optimize the padding
        seq = np.append(seq, [0]*(maxlen-len(seq)))
    seq = seq[:maxlen]
    return seq


mrrs = []
r1s = []
r5s = []
r10s = []
weights = []

batchs = []

for i in range(len(testset["query_array"]) // batch_size):
    queryVec = testset["query_array"][i*batch_size:(i+1)*batch_size]
    queryLen = testset["query_lenpos"][i*batch_size:(i+1)*batch_size]
    codeStr = testset["raw_code"][i*batch_size:(i+1)*batch_size]

    count = len(queryVec)
    weights.append(count)

    queryVec = [pad_seq(i, 120) for i in queryVec]

    queryVec = torch.LongTensor(queryVec).t()
    maxQueryLen = queryVec.max(axis=1).values.ne(
        0).sum().item()  # 当前batch中所有query的最大长度
    attention_mask = torch.LongTensor(queryVec).data.eq(
        0).t()[:, :maxQueryLen]  # [batch_size x maxQueryLen]
    attention_mask = attention_mask.cuda()
    if hasattr(ke, 'decoder') and hasattr(ke.decoder, 'attn'):
        ke.decoder.attn.applyMask(attention_mask)
    descGenVec = ke.translate([(torch.LongTensor(queryVec).cuda(), torch.LongTensor(queryLen).cuda(
    )), None, torch.LongTensor([[BOS]*count]).cuda(), None, None, None], 120)  # [max_desc_len x batch_size]
    queryVec = queryVec.t().numpy()

    descGenVec = [i.cpu().numpy() for i in descGenVec.t()]
    descGenStr = [' '.join([discri_dict[j] for j in i]) for i in descGenVec]
    queryStr = [' '.join([discri_dict[j] for j in i]) for i in queryVec]

    result = eval_integrate_easy_to_use(
        descGenStr, queryStr, codeStr, pool_size=poolsize, beta=beta,verbose=True)
    mrrs.append(result["mrr"])
    r1s.append(result["r1"])
    r5s.append(result["r5"])
    r10s.append(result["r10"])

print({
    "r1": np.average(r1s, weights=weights),
    "r5": np.average(r5s, weights=weights),
    "r10": np.average(r10s, weights=weights),
    "mrr": np.average(mrrs, weights=weights)
})