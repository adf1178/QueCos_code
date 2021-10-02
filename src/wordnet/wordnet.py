import string
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# f = open("QUERIES_for_training.302-450","r")
# fout = open("QUERIES_for_training_Expanded.302-450","w", encoding="utf-8")
stop_words = set(stopwords.words("english"))

def qe(line):
    # if not line:
    #     break
    _line = line
    line = line.replace('\n', '')
    line = line.split(" ", 1)
    new_line = line[0]
    # line = ["i", "have", "a", "pen"]
    line[1] = line[1].lower()
    line[1] = line[1].translate(str.maketrans('', '', string.punctuation))
    # print(line)
    word_tokens = word_tokenize(line[1])
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    synonyms = []

    count = 0
    for x in filtered_sentence:

        for syn in wordnet.synsets(x):
            count = 0
            for l in syn.lemmas():
                if count < 1:
                    if l.name() not in synonyms:
                        synonyms.append(l.name())
                        count += 1

        count = 0

    synonyms_string = ' '.join(synonyms)
    # new_line=" ".join([str(new_line),synonyms_string])
    synonyms = []
    return _line + ' ' + synonyms_string
    #     fout.write(new_line)
    #     fout.write('\n')


    # f.close()
    # fout.close()
if __name__ == '__main__':
    while(1):
        print(qe(input()))