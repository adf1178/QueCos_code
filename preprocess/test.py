import json
import pandas as pd
import nltk
import re

string = """
StringWriter writer = new StringWriter();
IOUtils.copy(inputStream, writer, encoding);
String theString = writer.toString();
"""
tokenizer = nltk.tokenize.WordPunctTokenizer()


def tokenize(string):
    string = re.sub('(?!^)([A-Z][a-z]+)', r' \1', string)
    string = re.sub('_', r' ', string)
    string = re.sub('([\[\]\(\){}\.,:;/\\=\'\"])', r'', string)
    string = re.sub('>>>', r' ', string)
    string = string.lower()

    return tokenizer.tokenize(string)


result = []
df = pd.read_csv(
    "/data/home/zhnong/preprocess/python_top_query 1-5-checked-new.csv", header=None, encoding="gbk")
for line in df[[1, 4]].values:
    dic = {}
    query = tokenize(line[0])
    code = tokenize(line[1])
    dic['query_tokens'] = query
    dic['code_tokens'] = code
    result.append(dic)
open('/data/home/zhnong/preprocess/python_top100.json',
     "w").write(json.dumps(result))
