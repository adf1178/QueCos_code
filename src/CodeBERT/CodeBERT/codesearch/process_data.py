# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gzip
import os
import json
import numpy as np
from more_itertools import chunked
# from tqdm import tqdm

DATA_DIR='../data/codesearch'

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def preprocess_test_data(language, test_batch_size=1000):
    path = os.path.join(DATA_DIR, '{}_test_0.jsonl'.format(language))
    print(path)
    with open(path, 'r') as pf:
        data = pf.readlines()

    idxs = np.arange(len(data))
    data = np.array(data, dtype=np.object)

    np.random.seed(0)   # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)
    batched_data = [i for i in batched_data]

    print("start processing")
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break # the last batch is smaller than the others, exclude.
        examples = []
        for d_idx, d in enumerate(batch_data):
            line_a = json.loads(d)
            doc_token = ' '.join(line_a['docstring_tokens'])
            for dd in batch_data:
                line_b = json.loads(dd)
                code_token = ' '.join([format_str(token) for token in line_b['code_tokens']])

                example = (str(1), line_a['url'], line_b['url'], doc_token, code_token)
                example = '<CODESPLIT>'.join(example)
                examples.append(example)

        data_path = os.path.join(DATA_DIR, 'test/{}'.format(language))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, 'batch_{}.txt'.format(batch_idx))
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))

if __name__ == '__main__':
    languages = ['java']
    for lang in languages:
        preprocess_test_data(lang)
