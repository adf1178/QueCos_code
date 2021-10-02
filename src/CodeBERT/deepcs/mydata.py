import os
import sys
import torch
import torch.utils.data as data
import torch.nn as nn
import tables
import json
import random
import numpy as np
import pickle
split = 400000


class mydataset(data.Dataset):
    def __init__(self, max_name_len, max_tok_len, max_desc_len, max_api_len):
        self.max_name_len = max_name_len
        self.max_tok_len = max_tok_len
        self.max_desc_len = max_desc_len
        self.api_len = max_api_len
        print("loading data ... ... ...")
        # self.trainset = np.load(os.path.join(os.path.dirname(
        # __file__), "./CSN-full/train.npy"), allow_pickle=True).item()
        self.trainset = np.load(
            "/data/home/zhnong/ke/src/deepcs/self/valid_allarray.npz", allow_pickle=True)
        self.names = self.trainset["name_array"]
        self.idx_names = self.trainset["name_lenpos"]

        self.apis = np.zeros((30)).astype('int64')

        self.desc = self.trainset["descri_array"]
        self.idx_descri = self.trainset["descri_lenpos"]

        self.token = self.trainset["token_array"]
        self.idx_token = self.trainset["token_lenpos"]
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

        rand_offset = random.randint(0, self.data_len-1)
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


class mydataset_valid(data.Dataset):
    def __init__(self, max_name_len, max_tok_len, max_desc_len, max_api_len):
        self.max_name_len = max_name_len
        self.max_tok_len = max_tok_len
        self.max_desc_len = max_desc_len
        self.api_len = max_api_len
        print("loading data ... ... ...")
        self.validset = np.load(os.path.join(os.path.dirname(
            __file__), "../../data-java/valid.npy"), allow_pickle=True).item()
        self.names = self.validset["name_array"]
        self.idx_names = self.validset["name_lenpos"]

        self.apis = np.zeros((30)).astype('int64')

        self.desc = self.validset["descri_array"]
        self.idx_descri = self.validset["descri_lenpos"]

        self.token = self.validset["token_array"]
        self.idx_token = self.validset["token_lenpos"]
        # self.data_len = self.names.shape[0]
        self.data_len = len(self.desc)
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


class mydataset_test(data.Dataset):
    def __init__(self, max_name_len, max_tok_len, max_desc_len, max_api_len):
        self.max_name_len = max_name_len
        self.max_tok_len = max_tok_len
        self.max_desc_len = max_desc_len
        self.api_len = max_api_len
        print("loading data ... ... ...")
        # self.testset = np.load(os.path.join(os.path.dirname(
        # __file__), "../../data-python4csn-ocor/test.npy"), allow_pickle=True).item()
        self.testset = np.load(os.path.join(
            "/data/czwang/ke/data-java/test.npy"), allow_pickle=True).item()
        # self.testset = np.load(os.path.join(
        #     "/data/home/zhnong/ke/data-python4csn/test.npy"), allow_pickle=True).item()
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


class mydataset_wordnet(data.Dataset):
    def __init__(self, max_name_len, max_tok_len, max_desc_len, max_api_len):
        self.max_name_len = max_name_len
        self.max_tok_len = max_tok_len
        self.max_desc_len = max_desc_len
        self.api_len = max_api_len
        print("loading data ... ... ...")
        self.testset = np.load(os.path.join(os.path.dirname(
            __file__), "../../data-python4csn/test.npy"), allow_pickle=True).item()
        # self.testset = np.load(
        #     "/data/home/zhnong/deepcs/DCS-master/pytorch/self/test_allarray.npz", allow_pickle=True)
        # self.names = self.testset["name_array"]
        # self.idx_names = self.testset["name_lenpos"]

        self.names = self.testset["name_array"]
        self.idx_names = self.testset["name_lenpos"]

        desc_dict = json.loads(
            open('/data/home/zhnong/ke/data-python4csn/descri.json').read())
        desc_dict2 = {desc_dict[i]: i for i in desc_dict}

        self.apis = np.zeros((30)).astype('int64')

        self.desc = []
        for i in self.testset["query_array"]:
            import string
            from nltk.corpus import wordnet
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords

            stop_words = set(stopwords.words("english"))

            line = [desc_dict2[j] for j in i]

            line[1] = line[1].lower()
            line[1] = line[1].translate(
                str.maketrans('', '', string.punctuation))
            word_tokens = word_tokenize(line[1])
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            synonyms = []

            for x in filtered_sentence:
                count = 0
                for syn in wordnet.synsets(x):
                    for l in syn.lemmas():
                        if(count < 3):
                            if l.name() not in synonyms:
                                synonyms.append(l.name())
                                count += 1
            synonyms = (list(i) + [desc_dict[j] if j in desc_dict else desc_dict['unk']
                                   for j in synonyms])[:max_desc_len]
            self.desc.append(synonyms)
        # self.idx_descri = self.testset["descri_lenpos"]

        # self.desc=self.testset["query_array"]
        # self.idx_descri=self.testset["query_lenpos"]

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


def load_dict(filename):
    return json.loads(open(filename, "r").readline())


# vocab_name = load_dict(os.path.join(os.path.dirname(__file__), '../../data/name.json'))
# vocab_tokens = load_dict(os.path.join(os.path.dirname(__file__),'../../data/token.json'))
# vocab_desc = load_dict(os.path.join(os.path.dirname(__file__),'../../data/descri.json'))


def indexes2sent(indexes, vocab, ignore_tok=0):
    '''indexes: numpy array'''

    def revert_sent(indexes, ivocab, ignore_tok=0):
        indexes = filter(lambda i: i != ignore_tok, indexes)
        toks, length = [], 0
        for idx in indexes:
            toks.append(ivocab.get(idx, '<unk>'))
            length += 1

        return ' '.join(toks), length

    ivocab = {v: k for k, v in vocab.items()}
    if indexes.ndim == 1:  # one sentence
        return revert_sent(indexes, ivocab, ignore_tok)
    else:  # dim>1
        sentences, lens = [], []  # a batch of sentences
        for inds in indexes:
            sentence, length = revert_sent(inds, ivocab, ignore_tok)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens

# train_set = mydataset_valid(6,50,30,30)
# # print(len(train_set))
# # # # print(train_set[179])
# train_data_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=1)
# print('============ Train Data ================')
# k=0
# for batch in train_data_loader:
#     batch = tuple([t.numpy() for t in batch])
#     tokens, tok_len, good_desc, good_desc_len, bad_desc, bad_desc_len = batch
#     k+=1
#     print(k)
#     if k>20: break
#     print('-------------------------------')
#     print(indexes2sent(tokens, vocab_tokens))
#     # print(apiseq.dtype,api_len)
#     # print(tokens.dtype,tok_len)
#     # print(good_desc.dtype,good_desc_len)
#     # print(bad_desc.dtype,bad_desc_len)
