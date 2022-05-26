import enum
import os.path
import json
from torchtext.legacy import data as torchdata
import torchtext.vocab as vocab
import torch

from Datasets.ThDataset import ThDataset
from imblearn.over_sampling import RandomOverSampler
import random
from collections import defaultdict
import re

class ThMDMCDataset(ThDataset):
    def __init__(self, dataset, tokenizer, **kwargs):
        super().__init__(dataset.basedir, dataset.name)

        self.window = None       if "window" not in kwargs else kwargs["window"]
        self.resampling = False  if "resampling" not in kwargs else kwargs["resampling"]
        self.seed = 42           if "seed" not in kwargs else kwargs["seed"]
        self.tokenizer = tokenizer

    def load_data(self, datasets, cache=True):
        window = self.window
        resampling = self.resampling
        seed = self.seed


        spopt = ""
        if window is not None:
            spopt += "-w"+str(window)
        
        if resampling:
            spopt += "-r"

        for sp in datasets.datasets:
            if cache:
                d, success = self._load_tmp_data(sp+spopt)
                if success:
                    self.datasets[sp] = d
                    continue

            dataset = datasets.datasets[sp]
            dataset = self._toshorten(dataset)
            nd = self._totoken(dataset)

            if sp=="train":
                if window is not None:
                    nd = self._slide(nd, window)

                if resampling:
                    nd = self._resample(nd)

            self.datasets[sp] = nd

            if cache:
                self._save_tmp_data(sp, nd)
        

    def _resample(self, dataset):
        # TODO: check validity
        assert(False)
        oversample = RandomOverSampler(sampling_strategy="all", random_state=self.seed)
        X = []
        Y = []
        
        N = len(dataset)
        cc = defaultdict(int)
        for i, d in enumerate(dataset):
            cnt = sum([l=="misp" for l in d["labels"]])/len(d["labels"])
            cnt = round(cnt, 1)
            X.append([i])
            Y.append(str(cnt))
            cc[cnt] += 1
    
        s = 0
        for p in cc:
            s += cc[p]*p
        X_over, Y_over = oversample.fit_resample(X, Y)
        newdataset = []


        random.seed(self.seed)
        idx = list(range(len(X_over)))
        random.shuffle(idx)
        for i in idx:
            x = X_over[i]
            y = Y_over[i]
            newdataset.append(dataset[x[0]])

            if len(newdataset) >= len(dataset):
                break

        return newdataset[0:len(dataset)]
        
    def _slide(self, dataset, window):
        # TODO: check validity
        assert(False)
        newdataset = []
        random.seed(self.seed)

        for d in dataset:
            if len(d["misp"]) <= window:
                newdataset.append(d)
            else:
                nwindow = len(d["misp"])-window+1
                for i in range(nwindow):
                    if random.random() > 1.0/nwindow :
                        continue

                    newdataset.append({
                        "misp": d["misp"][i:i+window],
                        "labels": d["labels"][i:i+window],
                        "corr": []
                    })
                    
            # Avoid Memory overflow
            if len(newdataset) > len(dataset):
                break

        random.shuffle(newdataset)

        return newdataset[0:len(dataset)]

    def _toshorten(self, dataset):
        newdataset = []
        repregx = re.compile(r'(.)\1{2,}')
        for d in dataset:
            newd = {"misp": [], "labels": [], "corr": []}
            for w, c, l in zip(d["misp"], d["corr"], d["labels"]):
                if w.strip()=="":
                    newdataset.append(newd)
                    newd = {"misp": [], "labels": [], "corr": []}
                    continue
                

                nw = repregx.sub(r'\1', w)
                # groups = [list(s) for _, s in groupby(w)]

                newd["misp"].append(nw)
                newd["labels"].append(l)
                newd["corr"].append(c)
            
            if len(newd["misp"])!=0:
                newdataset.append(newd)
            
        return newdataset

    def _totoken(self, dataset):
        return dataset

    def _get_columns(self, TOKEN, LABEL):
        return {
            "misp": TOKEN,
            "labels": LABEL,
            "corr": TOKEN
        }

    def get_config(self):
        return {
            "kwargs": {
                "cache": False,
                "window" : self.window
            },
            "decorator": True,
            "module": "Data.ThDatasetDecorator",
            "class": "ThDatasetDecorator",
        }
    
    def unk_replacement(self, sent, tokens, labels=None, probs=None):
        unk = self.fields["tokens"].unk_token
        if unk not in tokens:
            return tokens, labels, probs

        newtokens = []
        newlabels = []
        newprobs = []
        if labels is None:
            labels = ["" for t in tokens]
        
        if probs is None:
            probs = [[1] for t in tokens]

        # merge consecutive <unk>
        for tidx, token in enumerate(tokens):
            
            if token==unk and len(newtokens)>0 and newtokens[-1]==unk:
                p = max(newprobs[-1][0], probs[tidx][0])
                newprobs[-1] = [p]

                if p > 0.5:
                    newlabels[-1] = self.misp_token
                else:
                    newlabels[-1] = self.corr_token

                continue
                
            newtokens.append(token)
            newlabels.append(labels[tidx])
            newprobs.append(probs[tidx])
        
        nounk = []
        currsent = sent
        for tidx, token in enumerate(newtokens):
            if token==unk:
                prevsent = "".join(nounk)
                
                if tidx+1 < len(newtokens):
                    u = ""
                    initsent = currsent
                    while len(currsent) > 0 and (not currsent.startswith(newtokens[tidx+1])):
                        c = currsent[0]
                        u += c
                        currsent = currsent[1:]
                        # if len(currsent)==0:
                        #     print(sent)
                        #     print(initsent)
                        #     print(newtokens)
                        #     print(tidx)
                        #     assert(False)

                    nounk.append(u)
                        
                else:
                    nounk.append(currsent)        

            # special tokens
            elif "<" in token:
                nounk.append(token)  
            else:
                currsent = currsent[len(token):]
                nounk.append(token)  
                  
        return nounk, newlabels, newprobs

    