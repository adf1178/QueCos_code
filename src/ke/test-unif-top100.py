import os
import random
import torch
from ..deepcs.validate import validate, model,  config, validate_for_segments
from ..ke import lib
import torch.utils.data as data
import numpy as np
import torch

ke_path = os.path.join(os.path.dirname(
    __file__), '../../save/model_rf_hasBaselineke_python_csn_deepcs_mrr_reinforce/model_rf_hasBaselineke_python_csn_deepcs_mrr_reinforce_49.pt')


class mydataset_test(data.Dataset):
    def __init__(self, max_name_len, max_tok_len, max_desc_len, max_api_len):
        self.max_name_len = max_name_len
        self.max_tok_len = max_tok_len
        self.max_desc_len = max_desc_len
        self.api_len = max_api_len
        print("loading data ... ... ...")
        self.testset = np.load(os.path.join(os.path.dirname(
            __file__), "../../data-python4csn/test_clean.npy"), allow_pickle=True).item()
        self.names = self.testset["name_array"]
        self.idx_names = self.testset["name_lenpos"]

        self.apis = np.zeros((30)).astype('int64')

        # self.desc = self.testset["descri_array"]
        # self.idx_descri = self.testset["descri_lenpos"]

        self.desc = self.testset["query_array"]
        self.idx_descri = self.testset["query_lenpos"]

        self.token = self.testset["token_array"]
        self.idx_token = self.testset["token_lenpos"]
        self.data_len = len(self.desc)
        # self.data_len = self.names.shape[0]
        print("{} entries".format(self.data_len))

    def pad_seq(self, seq, maxlen):
        if len(seq) < maxlen:
            # !!!!! numpy appending is slow. Try to optimize the padding
            seq = np.append(seq, [0]*(maxlen-len(seq)))
        seq = seq[:maxlen]
        return seq

    def __len__(self):
        return self.data_len

    def __getitem__(self, offset):
        name = np.array(self.names[offset]).astype('int64')
        name_len = min(len(name), self.max_name_len)
        name_len = max(name_len, 1)
        name = self.pad_seq(name, self.max_name_len)

        good_desc = np.array(self.desc[offset]).astype('int64')
        good_desc_len = min(len(good_desc), self.max_desc_len)
        good_desc_len = max(good_desc_len, 1)
        good_desc = self.pad_seq(good_desc, self.max_desc_len)

        rand_offset = random.randint(0, self.data_len - 1)
        bad_desc = np.array(self.desc[rand_offset]).astype('int64')
        bad_desc_len = min(len(bad_desc), self.max_desc_len)
        bad_desc_len = max(bad_desc_len, 1)
        bad_desc = self.pad_seq(bad_desc, self.max_desc_len)

        token = np.array(self.token[offset]).astype('int64')
        token_len = min(len(token), self.max_tok_len)
        token_len = max(token_len, 1)
        token = self.pad_seq(token, self.max_tok_len)

        api = self.apis
        api_len = 1

        return name, name_len, token, token_len, good_desc, good_desc_len, bad_desc, bad_desc_len


if __name__ == "__main__":
    load = torch.load(ke_path)
    ke = None
    ke = load['model']
    # this is the test function, you can modify pool size and VALID or TEST set
    data = []
    data2 = []
    set = mydataset_test(6, 50, 30, 30)
    for i in set:
        queryVec = torch.LongTensor([i[4]])
        if ke is not None:
            # 使用ke对query进行处理
            if len(queryVec) < 120:
                queryVec = list(queryVec[0]) + \
                    [0 for i in range(120 - len(queryVec))]
                queryVec = torch.LongTensor(np.array([queryVec]))
            queryVec = queryVec.t()
            maxQueryLen = queryVec.max(axis=1).values.ne(
                0).sum().item()  # 当前batch中所有query的最大长度
            attention_mask = torch.LongTensor(queryVec).data.eq(
                0).t()[:, :maxQueryLen]  # [batch_size x maxQueryLen]
            attention_mask = attention_mask.cuda()
            ke.decoder.attn.applyMask(attention_mask)
            descVec, _ = ke.sample([(queryVec.cuda(), torch.LongTensor([maxQueryLen]).cuda()), None, torch.LongTensor(
                [[0]*1000]).cuda(), None, None, None], 120)  # [max_desc_len x batch_size]
            data.append((i[0], i[1], i[2], i[3], descVec.t().cpu().numpy()[
                        0], len(descVec.t()[0]), i[4], i[5]))
        data2.append(i)
    testresult = validate_for_segments(data, model, 1000,
                                       1, 'cos_integrate', 0.6)
    testresult2 = validate_for_segments(data2, model, 1000,
                                        1, 'cos')
    with open(os.path.join(os.path.dirname(__file__), '../../data-python4csn/descri.json')) as f:
        import json
        labelToIdx = json.loads(f.read())
        dict2 = {labelToIdx[i]: i for i in labelToIdx}
    for i in range(len(testresult["mrrs"])):
        if testresult["mrrs"][i] > testresult2["mrrs"][i] + 0.1:
            queryVec = set[i][4]
            desGenVec = data[i][4]
            print(
                '{} -> {}'.format(testresult2["mrrs"][i], testresult["mrrs"][i]))
            print(' '.join(
                [dict2[j] if j in dict2 else '<unk>' for j in queryVec if not j == 0]))
            print(' '.join(
                [dict2[j] if j in dict2 else '<unk>' for j in desGenVec if not j == 0]))
    del testresult["mrrs"]
    del testresult2["mrrs"]
    print(testresult)
    print(testresult2)
    print("done")
