import spacy
import re

class tokenize(object):
    
    def __init__(self, lang):
        # self.nlp = spacy.load(lang)
        if lang == 'en':
            from spacy.lang.en import English
            self.nlp = English()
        elif lang == 'fr':
            from spacy.lang.fr import French
            self.nlp = French()
            
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
