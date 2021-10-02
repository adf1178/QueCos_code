import os
import numpy as np
import torch
import json
from ..CodeBERT.CodeBERT.codesearch.demo.demo import eval_integrate_easy_to_use
from ..wordnet.wordnet import qe

poolsize = 1000
batch_size = 1000
beta=1.0
testset = np.load(os.path.join(os.path.dirname(__file__),
                               "../../data-python4csn/test.npy"), allow_pickle=True).item()
with open(os.path.join(os.path.dirname(__file__), '../../data-python4csn/descri.json')) as f:
    labelToIdx = json.loads(f.read())
    discri_dict = {labelToIdx[i]: i for i in labelToIdx}
#######################
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
    queryStr = [' '.join([discri_dict[j] for j in i]) for i in queryVec]

    descGenStr = [qe(i) for i in queryStr]
    count = len(queryVec)
    weights.append(count)

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