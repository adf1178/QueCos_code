import argparse
import os
import numpy as np
from . import lib
import torch

parser = argparse.ArgumentParser(description='')
parser.add_argument('-alpha', default=1.0, type=float)
opt = parser.parse_args()

print(opt.alpha)

poolsize = 1000
batch_size = 1000
alpha = opt.alpha
testset = np.load(os.path.join(os.path.dirname(__file__),
                               "../../data-python4staqc/test.npy"), allow_pickle=True).item()
ke_path = os.path.join(os.path.dirname(
    __file__), '../../save/model_rf_hasBaselinepython_staqc_mrr_ocor_reinforce/model_rf_hasBaselinepython_staqc_mrr_ocor_reinforce_40.pt')

###############################################################

load = torch.load(ke_path)
ke = load['model']


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
    codeVec = testset["token_array"][i*batch_size:(i+1)*batch_size]

    count = len(queryVec)
    weights.append(count)

    queryVec = [pad_seq(i, 120) for i in queryVec]

    queryVec = torch.LongTensor(queryVec).t()
    maxQueryLen = queryVec.max(axis=1).values.ne(
        0).sum().item()  # 当前batch中所有query的最大长度
    attention_mask = torch.LongTensor(queryVec).data.eq(
        0).t()[:, :maxQueryLen]  # [batch_size x maxQueryLen]
    attention_mask = attention_mask.cuda()
    ke.decoder.attn.applyMask(attention_mask)
    descGenVec, _ = ke.sample([(torch.LongTensor(queryVec).cuda(), torch.LongTensor(queryLen).cuda(
    )), None, torch.LongTensor([[0]*count]).cuda(), None, None, None], 120)  # [max_desc_len x batch_size]
    queryVec = queryVec.t().numpy()

    descGenVec = [i.cpu().numpy() for i in descGenVec.t()]

    data = []
    for j in range(len(descGenVec)):
        code = np.array(codeVec[j])
        descGen = descGenVec[j]
        query = queryVec[j]
        if descGen[0] == lib.data.BOS:
            descGen = descGen[1:]
        if query[0] == lib.data.BOS:
            query = query[1:]
        if len(np.where(descGen == lib.data.Constants.EOS)[0]):
            descGen = descGen[:np.where(
                descGen == lib.data.Constants.EOS)[0][0]]
        if len(np.where(query == lib.data.Constants.EOS)[0]):
            query = query[:np.where(query == lib.data.Constants.EOS)[0][0]]

        code[code == 3] = 1
        descGen[descGen == 3] = 1
        query[query == 3] = 1

        data.append((list(code), list(descGen), list(query)))
        # _result = eval_for_segments_easy_to_use(
        #     poolsize, [(code, descGen, query)], alpha=alpha, verbose=False)
        # print(_result['mrr'])
    batchs.append(data)
del ke
del load
torch.cuda.empty_cache()
for data in batchs:
    from ..OCoR.run import eval_for_segments_easy_to_use
    result = eval_for_segments_easy_to_use(
        poolsize, data, alpha=alpha, verbose=True)
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
