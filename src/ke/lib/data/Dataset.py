from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

from . import Constants


class Dataset(object):
    def __init__(self, data, data_name, batchSize, cuda, eval=False):
        self.data_name = data_name
        self.token = [torch.LongTensor(i) for i in data["token_array"]]
        self.src = [torch.LongTensor(i) for i in data["query_array"]]
        self.tgt = [torch.LongTensor(i) for i in data["descri_array"]]
        self.name = [torch.LongTensor(i) for i in data["name_array"]]
        self.raw_code = [None for _ in data["name_array"]]
        if 'raw_code' in data:
            self.raw_code = [i for i in data["raw_code"]]
        assert(len(self.src) == len(self.tgt))
        self.cuda = cuda

        self.batchSize = batchSize
        # self.numBatches = int(math.ceil(len(self.src)/batchSize)-1)
        self.numBatches = int(math.ceil(len(self.src) * 1.0 / batchSize))
        self.eval = eval

    def shuffle(self):
        data = list(zip(self.src, self.tgt, self.token, self.name, self.raw_code))
        random.shuffle(data)
        if data != []:
            print("good")
            self.src, self.tgt, self.token, self.name, self.raw_code = zip(*data)

    def _batchify(self, data, align_right=False, include_lengths=False):
        length_limit = 120
        lengths = [min(x.size(0), length_limit) for x in data]
        max_length = max(lengths)
        max_length = min(max_length, length_limit)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            data_length = min(data_length, max_length)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i][0:data_length])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        tokenBatch, token_lengths = self._batchify(
            self.token[index*self.batchSize:(index + 1)*self.batchSize], include_lengths=True)
        srcBatch, src_lengths = self._batchify(
            self.src[index*self.batchSize:(index + 1)*self.batchSize], include_lengths=True)
        tgtBatch = self._batchify(
            self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
        nameBatch, name_lengths = self._batchify(
            self.name[index * self.batchSize:(index + 1) * self.batchSize], include_lengths=True)
        raw_codeBatch = self.raw_code[index * self.batchSize:(index + 1) * self.batchSize]
        #qtBatch = self.qt[index * self.batchSize:(index + 1) * self.batchSize]

        indices = range(len(srcBatch))

        def wrap(b):
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            # b = Variable(b, volatile=self.eval)
            return b

        src_batch = zip(indices, srcBatch, tgtBatch, tokenBatch,
                        token_lengths, nameBatch, name_lengths, [None for i in indices], raw_codeBatch)
        src_batch, src_lengths = zip(
            *sorted(zip(src_batch, src_lengths), key=lambda x: -x[1]))
        indices, srcBatch, tgtBatch, tokenBatch, token_lengths, nameBatch, name_lengths, idxBatch, raw_codeBatch = zip(
            *src_batch)

        return (wrap(srcBatch), list(src_lengths)), \
            None, \
            wrap(tgtBatch), \
            indices, \
            None,\
            None,\
            wrap(tokenBatch),\
            token_lengths,\
            wrap(nameBatch),\
            name_lengths, \
            raw_codeBatch

    def __len__(self):
        return self.numBatches
