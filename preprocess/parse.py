import json
import numpy as np

from Dict import Dict

with open('./descri.json') as f:
    labelToIdx = json.loads(f.read())
    desc = Dict(labelToIdx)
with open('./token.json') as f:
    labelToIdx = json.loads(f.read())
    tokens = Dict(labelToIdx)
with open('./name.json') as f:
    labelToIdx = json.loads(f.read())
    name = Dict(labelToIdx)


def parse(type):
    with open('/data/home/zhnong/preprocess/python_top100.json'.format(type, type), "r+", encoding="utf8") as f:
        output = {}
        output['query_array'] = []
        output['query_lenpos'] = []
        # output['descri_array'] = []
        # output['descri_lenpos'] = []
        # output['name_array'] = []
        # output['name_lenpos'] = []
        output['token_array'] = []
        output['token_lenpos'] = []
        for item in json.load(f):
            output['query_array'].append(desc.convertToIdx(
                item['query_tokens'], 'unk', '<s>', '</s>').numpy())
            output['query_lenpos'].append(len(output['query_array'][-1]))

            # output['descri_array'].append(desc.convertToIdx(
            #     item['docstring_tokens'], '<nofd>', '<startt>', '</startt>').numpy())
            # output['descri_lenpos'].append(len(output['descri_array'][-1]))

            # output['name_array'].append(name.convertToIdx(
            #     item['func_name_tokens'], '<nofd>', '<startt>', '</startt>').numpy())
            # output['name_lenpos'].append(len(output['name_array'][-1]))

            output['token_array'].append(tokens.convertToIdx(
                item['code_tokens'], 'unk', '<s>', '</s>').numpy())
            output['token_lenpos'].append(len(output['token_array'][-1]))
        np.save('./top100-python.npy', output)


parse('train')
