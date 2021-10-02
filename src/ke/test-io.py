import os
import json
from pickle import TRUE
import torch

from .lib.data.Constants import BOS
from . import lib
import numpy as np

has_cuda = torch.cuda.is_available()

ke_path = os.path.join(os.path.dirname(
    __file__), '../../save/model_rf_hasBaselinepython_csn_mrr_code_bert_pretrain/model_rf_hasBaselinepython_csn_mrr_code_bert_pretrain_20.pt')

load = torch.load(ke_path)
ke = load['model']
ke.eval()
with open(os.path.join(os.path.dirname(__file__), '../../data-python4csn/descri.json')) as f:
    labelToIdx = json.loads(f.read())
    dict = lib.Dict(labelToIdx)
    dict2 = {labelToIdx[i]: i for i in labelToIdx}
while(1):
    query = input()
    queryVec = dict.convertToIdx(
        query.split(), "<nofd>", "<s>", "</s>").numpy()
    queryVecLen = queryVec.shape[0]
    attention_mask = torch.LongTensor(queryVec).data.eq(lib.Constants.PAD).t()
    if has_cuda:
        ke = ke.cuda()
        attention_mask = attention_mask.cuda()
    ke.decoder.attn.applyMask(attention_mask)
    if len(queryVec) < 120:
        queryVec = list(queryVec) + [0 for i in range(120 - len(queryVec))]
        queryVec = np.array(queryVec)
    queryVec = torch.LongTensor([[i] for i in queryVec])
    if has_cuda:
        queryVec = queryVec.cuda()

    count = len(queryVec)
    descGenVec = ke.translate([(queryVec.cuda(), torch.LongTensor([queryVecLen]).cuda(
    )), None, torch.LongTensor([[0]*count]).cuda(), None, None, None], 120)  # [max_desc_len x batch_size]
    # descVec, _ = ke.sample([(queryVec.cuda(), [queryVecLen]), None, torch.LongTensor(
    #     [[0]*1000]).cuda(), None, None, None], 120)  # [max_desc_len x batch_size]

    print(
        ' '.join([dict2[i[0]] or '<unk>' for i in descGenVec.cpu().detach().numpy()]))
