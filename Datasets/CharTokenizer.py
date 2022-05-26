from Datasets.BaseTokenizer import BaseTokenizer

class CharTokenizer(BaseTokenizer):
    def __init__(self, dataset, **kwargs):
        self.name = "char"

    def get_config(self):
        return {
            "kwargs": {},
            "module": "Datasets.CharTokenizer",
            "class": "CharTokenizer",
        }

    def tokenize(self, sents):
        tokens = []
        for s in sents:
            tokens.append(list(s))
        
        return tokens

    def tokenizeWithLabel(self, sents, sentLabels):
        tokens = []
        tokenLabels = []

        for words, labels in zip(sents, sentLabels):
            assert(len(words)==len(labels))

            _tmpTokens = []
            _tmpTokenLabels = []
            for w, l in zip(words, labels):
                _tmpTokens += list(w)
                _tmpTokenLabels += [l for _ in w]
            
            tokens.append(_tmpTokens)
            tokenLabels.append(_tmpTokenLabels)
        return tokens, tokenLabels