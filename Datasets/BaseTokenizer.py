class BaseTokenizer():
    def __init__(self, dataset, **kwargs):
        self.name = "char"

    def tokenize(self, sents):
        assert(False)

    def get_config(self):
        return {
            "kwargs": {},
            "module": "Datasets.BaseTokenizer",
            "class": "BaseTokenizer",
        }