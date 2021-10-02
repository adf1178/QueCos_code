from nltk.tokenize import word_tokenize
import pickle
import os
import numpy as np
import sys
import random

from tqdm import tqdm
from .ModelWrapper import *
from .vocab import *
from .config import dataname

# use shared vocabulary


class DataSet:
    def __init__(self, config, dataName="train"):
        self.train_path = os.path.join(os.path.dirname(
            __file__), "data/{}/train.txt".format(dataname))
        self.val_path = os.path.join(os.path.dirname(
            __file__), "data/{}/valid.txt".format(dataname))
        self.val_s_path = os.path.join(os.path.dirname(
            __file__), "data/{}/valid_s.txt".format(dataname))
        self.test_path = os.path.join(os.path.dirname(
            __file__), "data/{}/test.txt".format(dataname))
        self.dev_path = os.path.join(os.path.dirname(
            __file__), "data/{}/dev.txt".format(dataname))
        self.eval_path = os.path.join(os.path.dirname(
            __file__), "data/{}/eval.txt".format(dataname))
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.Nls = []
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/nl_voc.pkl".format(dataname))):
            self.init_dic()
        self.Load_Voc()
        if dataName == "train":
            if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/data.pkl".format(dataname))):
                self.data = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/data.pkl".format(dataname)), "rb"))
                self.Nls = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/Nls.pkl".format(dataname)), "rb"))
                self.Codes = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/Codes.pkl".format(dataname)), "rb"))
                return
            self.data = self.preProcessData(
                open(self.train_path, "r", encoding='utf-8'))
        elif dataName == "val":
            if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/valdata.pkl".format(dataname))):
                self.data = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/valdata.pkl".format(dataname)), "rb"))
                self.Nls = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/valNls.pkl".format(dataname)), "rb"))
                self.Codes = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/valCodes.pkl".format(dataname)), "rb"))
                return
            self.data = self.preProcessData(
                open(self.val_path, "r", encoding='utf-8'))
        elif dataName == "val_s":
            if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/val_s_data.pkl".format(dataname))):
                self.data = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/val_s_data.pkl".format(dataname)), "rb"))
                self.Nls = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/val_s_Nls.pkl".format(dataname)), "rb"))
                self.Codes = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/val_s_Codes.pkl".format(dataname)), "rb"))
                return
            self.data = self.preProcessData(
                open(self.val_s_path, "r", encoding='utf-8'))
        elif dataName == "test":
            if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/testdata.pkl".format(dataname))):
                self.data = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/testdata.pkl".format(dataname)), "rb"))
                self.Nls = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/testNls.pkl".format(dataname)), "rb"))
                self.Codes = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/testCodes.pkl".format(dataname)), "rb"))
                return
            self.data = self.preProcessData(
                open(self.test_path, "r", encoding='utf-8'))
        elif dataName == "dev":
            if os.path.exists("data/{}/devdata.pkl"):
                self.data = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/devdata.pkl".format(dataname)), "rb"))
                self.Nls = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/devNls.pkl".format(dataname)), "rb"))
                self.Codes = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/devCodes.pkl".format(dataname)), "rb"))
                return
            self.data = self.preProcessData(
                open(self.dev_path, "r", encoding='utf-8'))
        else:
            if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/evaldata.pkl".format(dataname))):
                self.data = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/evaldata.pkl".format(dataname)), "rb"))
                self.Nls = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/evalNls.pkl".format(dataname)), "rb"))
                self.Codes = pickle.load(open(os.path.join(os.path.dirname(
                    __file__), "data/{}/evalCodes.pkl".format(dataname)), "rb"))
                return
            self.data = self.preProcessData(
                open(self.eval_path, "r", encoding='utf-8'))

    def Load_Voc(self):
        if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/nl_voc.pkl".format(dataname))):
            self.Nl_Voc = pickle.load(open(os.path.join(os.path.dirname(
                __file__), "data/{}/nl_voc.pkl".format(dataname)), "rb"))
        if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/code_voc.pkl".format(dataname))):
            self.Code_Voc = pickle.load(open(os.path.join(os.path.dirname(
                __file__), "data/{}/code_voc.pkl".format(dataname)), "rb"))
        if os.path.exists(os.path.join(os.path.dirname(__file__), "data/{}/char_voc.pkl".format(dataname))):
            self.Char_Voc = pickle.load(open(os.path.join(os.path.dirname(
                __file__), "data/{}/char_voc.pkl".format(dataname)), "rb"))

    def init_dic(self):
        print("initVoc")
        f = open(self.train_path, "r", encoding='utf-8')
        lines = f.readlines()
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for i in range(int(len(lines) / 2)):
            Nl = lines[2 * i].strip()
            Code = lines[2 * i + 1].strip()
            Nl_tokens = Nl.split()
            Code_Tokens = Code.split()
            Nls.append(Nl_tokens)
            # Nls.append(Code_Tokens)
            Codes.append(Code_Tokens)
            maxNlLen = max(maxNlLen, len(Nl_tokens))

            maxCodeLen = max(maxCodeLen, len(Code_Tokens))
        # print(Nls)
        # print("------------------")
        # nl_voc = VocabEntry.from_corpus(Nls, size=10000, freq_cutoff=3)
        # code_voc = VocabEntry.from_corpus(Codes, size=10000, freq_cutoff=3)
        # self.Nl_Voc = nl_voc.word2id
        # self.Code_Voc = code_voc.word2id
        import json
        self.Nl_Voc = json.load(open(os.path.join(os.path.dirname(__file__),"data/{}/descri.json".format(dataname))))
        self.Code_Voc = json.load(open(os.path.join(os.path.dirname(__file__),"data/{}/token.json".format(dataname))))

        for x in self.Nl_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        for x in self.Code_Voc:
            maxCharLen = max(maxCharLen, len(x))
            for c in x:
                if c not in self.Char_Voc:
                    self.Char_Voc[c] = len(self.Char_Voc)
        # open(os.path.join(os.path.dirname(__file__), "data/{}/nl_voc.pkl".format(dataname)), "wb").write(pickle.dumps(self.Nl_Voc))
        # open(os.path.join(os.path.dirname(__file__), "data/{}/code_voc.pkl".format(dataname)), "wb").write(pickle.dumps(self.Code_Voc))
        open(os.path.join(os.path.dirname(__file__), "data/{}/char_voc.pkl".format(dataname)),
             "wb").write(pickle.dumps(self.Char_Voc))
        # print(self.Nl_Voc)
        # print(self.Code_Voc)
        print(maxNlLen, maxCodeLen, maxCharLen)

    def Get_Em(self, WordList, NlFlag=True):
        ans = []
        for x in WordList:
            if NlFlag:
                if x not in self.Nl_Voc:
                    ans.append(1)
                else:
                    ans.append(self.Nl_Voc[x])
            else:
                if x not in self.Code_Voc:
                    ans.append(1)
                else:
                    ans.append(self.Code_Voc[x])
        return ans

    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans

    def get_overlap_indices(self, question, answer):
        a = []
        b = []
        ban = ['unk', 'Unknown', '<s>', '</s>']
        for x in question:
            isOverlap = False
            ma = 0
            for y in answer:
               #     ma = 0
                if x in y and x not in ban and y not in ban:
                    isOverlap = True
                    ma = max(ma, int(100 * (len(x) / len(y))))
                    # break
            a.append(ma)
            # if not isOverlap:
            #    a.append(0)
        for x in answer:
            isOverlap = False
            mb = 0
            for y in question:
                #mb = 0
                if x in y and x not in ban and y not in ban:
                    isOverlap = True
                    mb = max(mb, int(100 * (len(x) / len(y))))
                    # break
            b.append(mb)
            # if not isOverlap:
            #    b.append(0)
        a, _ = self.pad_seq(a, self.Nl_Len)
        b, _ = self.pad_seq(b, self.Code_Len)
        return a, b

    def preProcessData(self, datafile):
        lines = datafile.readlines()
        Nl_Sentences = []
        Code_Sentences = []
        Nl_Chars = []
        Code_Chars = []
        Nl_Overlap = []
        Code_Overlap = []
        res = []
        for i in tqdm(range(int(len(lines) / 2))):
            Nl = lines[2 * i].strip()
            Code = lines[2 * i + 1].strip()
            if len(Code) == 0:
                continue
            Nl_tokens = Nl.split()
            Code_Tokens = Code.split()
            self.Nls.append(Nl_tokens)
            self.Codes.append(Code_Tokens)
            Nl_Sentences.append(self.Get_Em(Nl_tokens))
            Code_Sentences.append(self.Get_Em(Code_Tokens, False))
            Nl_Chars.append(self.Get_Char_Em(Nl_tokens))
            Code_Chars.append(self.Get_Char_Em(Code_Tokens))
            res.append([0, 1])
            a, b = self.get_overlap_indices(Nl_tokens, Code_Tokens)
            Nl_Overlap.append(a)
            Code_Overlap.append(b)
        for i in tqdm(range(len(Nl_Sentences))):
            Nl_Sentences[i], _ = self.pad_seq(Nl_Sentences[i], self.Nl_Len)
            Code_Sentences[i], _ = self.pad_seq(
                Code_Sentences[i], self.Code_Len)
            for j in range(len(Nl_Chars[i])):
                Nl_Chars[i][j], _ = self.pad_seq(Nl_Chars[i][j], self.Char_Len)
            for j in range(len(Code_Chars[i])):
                Code_Chars[i][j], _ = self.pad_seq(
                    Code_Chars[i][j], self.Char_Len)
            Nl_Chars[i] = self.pad_list(
                Nl_Chars[i], self.Nl_Len, self.Char_Len)
            Code_Chars[i] = self.pad_list(
                Code_Chars[i], self.Code_Len, self.Char_Len)
        Nl_Sentences = np.array(Nl_Sentences, np.int32)
        Code_Sentences = np.array(Code_Sentences, np.int32)
        Nl_Chars = np.array(Nl_Chars, np.int32)
        Code_Chars = np.array(Code_Chars, np.int32)
        Nl_Overlap = np.array(Nl_Overlap, np.int32)
        Code_Overlap = np.array(Code_Overlap, np.int32)
        res = np.array(res)
        #Nl_Overlap = np.array(Nl_Overlap, np.int32)
        #Code_Overlap = np.array(Code_Overlap, np.int32)
        batchs = [Nl_Sentences, Nl_Chars, Code_Sentences,
                  Code_Chars, Nl_Overlap, Code_Overlap, res]
        if self.dataName == "train":
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/data.pkl".format(dataname)), "wb").write(pickle.dumps(batchs))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/Nls.pkl".format(dataname)), "wb").write(pickle.dumps(self.Nls))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/Codes.pkl".format(dataname)), "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "val_s":
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/val_s_data.pkl".format(dataname)), "wb").write(pickle.dumps(batchs))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/val_s_Nls.pkl".format(dataname)), "wb").write(pickle.dumps(self.Nls))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/val_s_Codes.pkl".format(dataname)), "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "val":
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/valdata.pkl".format(dataname)), "wb").write(pickle.dumps(batchs))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/valNls.pkl".format(dataname)), "wb").write(pickle.dumps(self.Nls))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/valCodes.pkl".format(dataname)), "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "test":
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/testdata.pkl".format(dataname)), "wb").write(pickle.dumps(batchs))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/testNls.pkl".format(dataname)), "wb").write(pickle.dumps(self.Nls))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/testCodes.pkl".format(dataname)), "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "dev":
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/devdata.pkl".format(dataname)), "wb").write(pickle.dumps(batchs))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/devNls.pkl".format(dataname)), "wb").write(pickle.dumps(self.Nls))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/devCodes.pkl".format(dataname)), "wb").write(pickle.dumps(self.Codes))
        if self.dataName == "eval":
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/evaldata.pkl".format(dataname)), "wb").write(pickle.dumps(batchs))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/evalNls.pkl".format(dataname)), "wb").write(pickle.dumps(self.Nls))
            open(os.path.join(os.path.dirname(
                __file__), "data/{}/evalCodes.pkl".format(dataname)), "wb").write(pickle.dumps(self.Codes))
        return batchs

    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq, act_len

    def pad_list(self, seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def Get_Train(self, batch_size, name="train"):
        data = self.data
        loaddata = []
        if self.dataName == "train":

            NegId = []
            for i in tqdm(range(len(data[0]))):
                tmp = []
                for j in range(5):
                    rand_offset = random.randint(0, len(data[0]) - 1)
                    while rand_offset == i:
                        rand_offset = random.randint(0, len(data[0]) - 1)
                    tmp.append(rand_offset)
                NegId.append(tmp)
            maxlen = len(data[0])
            tmp = []
            for i in range(len(data)):
                tmp.append([])
            for i in tqdm(range(maxlen)):
                for x in NegId[i]:
                    tmp[0].append(data[0][i])
                    tmp[1].append(data[1][i])
                    tmp[2].append(data[2][x])
                    tmp[3].append(data[3][x])
                    a, b = self.get_overlap_indices(self.Nls[i], self.Codes[x])
                    tmp[4].append(np.array(a, np.int32))
                    tmp[5].append(np.array(b, np.int32))
                    tmp[6].append([1, 0])
            for i in range(len(data)):
                loaddata.append(np.append(data[i], tmp[i], axis=0))
            shuffle = np.random.permutation(range(len(loaddata[0])))
            for i in range(len(data)):
                loaddata[i] = loaddata[i][shuffle]
        else:
            loaddata = data
        batch_nums = int(len(loaddata[0]) / batch_size)
        # print(batch_nums)
        for i in range(batch_nums):
            ans = []
            for j in range(len(loaddata)):
                ans.append(loaddata[j][batch_size * i:batch_size * (i + 1)])
            yield ans
            # yield Nl_Sentences[batch_size * i: batch_size * (i + 1)],Nl_Chars[batch_size * i: batch_size * (i + 1)],Code_Sentences[batch_size * i: batch_size * (i + 1)],Code_Chars[batch_size * i: batch_size * (i + 1)],Neg_Code_Sentences[batch_size * i: batch_size * (i + 1)], Neg_Code_Chars[batch_size * i: batch_size * (i + 1)]
