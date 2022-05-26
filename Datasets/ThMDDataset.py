from Datasets.ThMDMCDataset import ThMDMCDataset

class ThMDDataset(ThMDMCDataset):
    def __init__(self, dataset, tokenizer, **kwargs):
        super().__init__(dataset, tokenizer, **kwargs)

        self.tokenizer = tokenizer
        self.name = f"md-{dataset.name}-{tokenizer.name}"
        cache = True if "cache" not in kwargs else kwargs["cache"]

        self.load_data(dataset, cache)
    
    def _totoken(self, dataset):
        newdataset = []
        tokenizer = self.tokenizer
        
        sents = [d["misp"] for d in dataset]
        labels = [d["labels"] for d in dataset]

        sents, labels = tokenizer.tokenizeWithLabel(sents, labels)

        for didx, d in enumerate(dataset):
            newd = {"sent": "", "misp": [], "labels": [], "corr": []}
            newd["sent"] = "".join(d["misp"])
            
            newd["misp"].append(self.ssent_token)
            newd["labels"].append(self.corr_token)

            newd["misp"] += sents[didx]
            newd["labels"] += labels[didx]
            
            newd["misp"].append(self.esent_token)
            newd["labels"].append(self.corr_token)
            
            newdataset.append(newd)

        return newdataset
    
    def process(self, sent, device):
        spans = sent.split(" ")
        dataset = []
        for s in spans:
            dataset.append({
                "misp": [s],
                "corr": [s],
                "labels": [self.misp_token]
            })

        # TODO: implement window?
        dataset = self._toshorten(dataset)
        dataset = self._totoken(dataset)
        inputs = [d["misp"] for d in dataset]
        sents = [d["sent"] for d in dataset]

        return super().process(inputs, device), sents
        

    def _get_columns(self, TOKEN, LABEL):
        return {
            "misp": TOKEN,
            "labels": LABEL
        }

    def get_config(self):
        return {
            "kwargs": {
                "cache": False,
                "window" : self.window
            },
            "tokenizer": self.tokenizer.get_config(),
            "decorator": True,
            "module": "Datasets.ThMDDataset",
            "class": "ThMDDataset",
        }