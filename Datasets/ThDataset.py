
import os.path
import json
from attr import field
from torchtext.legacy import data as torchdata
from collections import Counter
from torchtext.legacy.vocab import Vocab
import torch
from collections import defaultdict

class ThDataset:
    def __init__(self, dir, name):
        self.basedir = dir
        self.name = name
        # self.rawdatasets = {}
        self.split = ["train", "test"]
        self.datasets = {}

        self.misp_token = "misp"
        self.corr_token = "corr"

        self.ssent_token = "<sent>"
        self.esent_token = "</sent>"
        self.unk_token = "<unk>"
        
        # determine later after calling build()
        self.batch_size = -1 
        self.fields = None
    
    def load_data(self, cache=True):
        rawdatasets = {}
        for sp in self.split:
            if cache:
                d, success = self._load_tmp_data(sp)
                if not success:
                    d = self.read(sp)
            else:
                d = self.read(sp)
            
            self._save_tmp_data(sp, d)
            rawdatasets[sp] = d
        nval = int(len(rawdatasets["train"])*0.1)
        nval = min(1000, nval)
        
        self.datasets = {
            "train": rawdatasets["train"][nval:],
            "validation": rawdatasets["train"][0:nval],
            "test": rawdatasets["test"],
        }

    def _load_tmp_data(self, sp):
        if os.path.exists(f'./Files/{self.name}_{sp}.jsonl'):
            return ThDataset.load_jsonl(f'./Files/{self.name}_{sp}.jsonl'), True

        return None, False
    
    def _save_tmp_data(self, sp, d):
        if not os.path.exists('./Files'):
            os.makedirs('Files')
        
        ThDataset.save_jsonl(f'./Files/{self.name}_{sp}.jsonl', d)
        return True

    @staticmethod
    def save_jsonl(filename, data):
        with open(filename, "w", encoding="utf-8") as fout:
            for d in data:
                json.dump(d, fout, ensure_ascii=False)
                fout.write('\n')
    
    @staticmethod
    def load_jsonl(filename):
        data = []
        with open(filename, encoding="utf-8") as fin:
            for line in fin:
                data.append(json.loads(line))
        return data
                

    def read(self, split):
        raise Exception("Not implemented")

    @staticmethod
    def _to_torch_example(d, FIELDS, TOKEN, LABEL):
        data = {}
        fields = {}
        for k, f in FIELDS:
            fields[k] = (k, f)
            data[k] = d[k]

        return torchdata.Example.fromdict(data, fields=fields)

    def save_fields(self, filename):
        if self.fields is None:
            print("Error: the dataset has not been built yet")
            assert(self.fields is not None)

        saved_data = []
        for k in self.fields:
            field = self.fields[k]

            specials = filter(None, [
                field.unk_token, 
                field.pad_token, 
                field.init_token,
                field.eos_token,
            ])

            specials = list(specials)
            data = {
                "field": k,
                "lower": field.lower,
                "counter": field.vocab.freqs,
                "specials": specials,

                "unk_token": field.unk_token,
                "pad_token": field.pad_token,
                "init_token": field.init_token,
                "eos_token": field.eos_token,
            }

            saved_data.append(data)
        
        return ThDataset.save_jsonl(filename, saved_data)

    
    def load_fields(self, filename):
        data = ThDataset.load_jsonl(filename)
        self.fields = {}

        for row in data:
            counter = Counter(row["counter"])
            vocab = Vocab(counter, specials=row["specials"])

            kwargs = {
                "lower": row["lower"],
                "unk_token": row["unk_token"],
                "pad_token": row["pad_token"],
                "init_token": row["init_token"],
                "eos_token": row["eos_token"],
            }

            newfield = torchdata.Field(**kwargs)
            newfield.vocab = vocab
            self.fields[row["field"]] = newfield
            
    def get_config(self):
        return {
            "kwargs": {},
            "module": "Data.ThDataset",
            "class": "ThDataset",
        }

    def _get_columns(self, TOKEN, LABEL):
        return {
            "misp": TOKEN,
            "labels": LABEL,
            "corr": TOKEN,
        }

    def build(self, device, MIN_FREQ = 1, BATCH_SIZE = 256, init_token = None, eos_token = None):
        self.batch_size = BATCH_SIZE
        TOKEN = torchdata.Field(lower = False, init_token = init_token, eos_token = eos_token)
        LABEL = torchdata.Field(unk_token = None)
        
        self.fields = {
            "tokens": TOKEN,
            "labels": LABEL
        }
        
        self.columns = self._get_columns(TOKEN, LABEL)

        self.torchdataset = {}
        fields = [(k, self.columns[k]) for k in self.columns]
        for sp in self.datasets:
            examples = [ThDataset._to_torch_example(d, fields, TOKEN, LABEL) for d in self.datasets[sp]]
            self.torchdataset[sp] = torchdata.Dataset(examples, fields=fields)

        TOKEN.build_vocab(
            self.torchdataset["train"], 
            # self.torchdataset["validation"], 
            # self.torchdataset["test"], 
            min_freq = MIN_FREQ,)

        # print(LABEL.vocab.itos)
        LABEL.build_vocab(self.torchdataset["train"])

        train_iterator, valid_iterator, test_iterator = torchdata.BucketIterator.splits(
            (self.torchdataset["train"], self.torchdataset["validation"], self.torchdataset["test"]), 
            sort = False,
            batch_size = BATCH_SIZE,
            device = device)

        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.test_iterator = test_iterator

    def process(self, inputs, device):
        TOKEN = self.fields["tokens"]
        LABEL = self.fields["labels"]

        columns = self._get_columns(TOKEN, LABEL)
        fields = [(k, columns[k]) for k in columns]
        examples = [
            ThDataset._to_torch_example({
                "misp": sent,
                "labels": [],
                "corr": []
            }, fields, TOKEN, LABEL)

            for sent in inputs
        ]

        torchdataset = torchdata.Dataset(examples, fields=fields)
        iterator = torchdata.BucketIterator(torchdataset, batch_size=1, device = device, shuffle=False)
        return iterator
    

    def describe(self):
        print("Dataset Info")
        nclass = None
        nacc = 0
        for split in self.datasets:
            print("Split:", split)
            rows = self.datasets[split]
            acc_length = 0
            max_length = 0
            cnt = defaultdict(int)
            acc_cnt = 0
            for d in rows:
                acc_length += len(d["misp"])
                max_length = max(max_length, len(d["misp"]))
                for l in d["labels"]:
                    cnt[l] += 1
                    acc_cnt += 1
            print(f"\t #Sents: {len(rows)}")
            print(f"\t #Tokens: {acc_length}")
            print(f"\t Avg Length: {acc_length/len(rows):.2f}")
            print(f"\t Max Length: {max_length}")
            print(f"\t Label: {dict(cnt)}")
            if cnt["corr"] > 0:
                print(f"\t Ratio: {cnt['misp']/cnt['corr']:.2f}")

            if split=="train":
                # class weighted = 1- cnt[k]/acc_cnt
                #                ~ (acc_cnt - cnt[k])
                nclass = cnt
                nacc = acc_cnt

        print("")
        return nacc, nclass
