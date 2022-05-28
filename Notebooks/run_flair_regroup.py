import sys
sys.path.append('../')

from flair.data import Dictionary
from flair.models import LanguageModel

from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus, TextDataset
import torch
import logging
log = logging.getLogger("flair")
import random


from Datasets.ThDatasetVISTEC import ThDatasetVISTEC
from util import *

import pickle

with open(f'./Models/merged_addition_tokens', 'rb') as f:
    additionTokens = pickle.load(f)


def transform_merging(sent, p):
    random.seed(42)
    tokens = list(sent)
    
    newtokens = []
    for i, t in enumerate(tokens):
        if random.random() < p and len(newtokens) > 0:
          if len(newtokens[-1]) >= 3:
            # No longer than 3
            newtokens.append(t)
          else:
            newtokens[-1] = newtokens[-1] + t
        else:
          newtokens.append(t)
    
    tokens = newtokens
    newtokens = []
    for t in tokens:
        if len(t) > 1:
            if t in additionTokens:
              newtokens.append(additionTokens[t])
            else:
              newtokens.extend(t)
        else:
            newtokens.append(t)
    return newtokens

from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus, TextDataset
import torch
import logging
log = logging.getLogger("flair")
import random

class MyTextDatasetReGroup(TextDataset):
    def __init__(
        self,
        dataset,
        dictionary: Dictionary,
        p_regroup: float, 
        expand_vocab: bool = False,
        forward: bool = True,
        split_on_char: bool = True,
        random_case_flip: bool = True,    #ignore
        document_delimiter: str = "\n", #ignore
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.dictionary = dictionary
        self.split_on_char = split_on_char
        self.forward = forward
        self.random_case_flip = random_case_flip
        self.expand_vocab = expand_vocab
        self.document_delimiter = document_delimiter
        self.shuffle = shuffle
        self.p_regroup = p_regroup

        self.files = []

    def __len__(self):
        return 1

    def __getitem__(self, index=0) -> torch.Tensor:
        """Tokenizes a text file on character basis."""
        lines = []
        for d in self.dataset:
            sent = d["sent"]
            chars = transform_merging(sent, p=self.p_regroup)
#             l = chars
            l = chars + [self.document_delimiter]
            lines.append(l)

        log.info(f"read text file with {len(lines)} lines")

        if self.shuffle:
            random.shuffle(lines)
            log.info("shuffled")

        if self.expand_vocab:
            for chars in lines:
                for char in chars:
                    self.dictionary.add_item(char)
                    
        nums = [self.dictionary.get_idx_for_item(char) for chars in lines for char in chars]
        ids = torch.tensor(nums, dtype=torch.long)
        
        if not self.forward:
            ids = ids.flip(0)
        return ids
    
class MyTextCorpus(TextCorpus):
    def __init__(
        self,
        datasets,
        dictionary: Dictionary,
        p_regroup: float,
        forward: bool = True,
        character_level: bool = True,
    ):
        self.dictionary: Dictionary = dictionary
        self.forward = forward
        self.split_on_char = character_level
        self.p_regroup = p_regroup
        
        
        self.random_case_flip = True
        self.expand_vocab = False
        self.document_delimiter = "\n"
        self.shuffle = True
        
        splitmaping = {
            "train": "train",
            "valid": "validation",
            "test": "test",
        }
        
        for k in splitmaping:
            sp = splitmaping[k]
            d = MyTextDatasetReGroup(
                datasets.datasets[sp],
                dictionary,
                p_regroup = self.p_regroup,
                forward = self.forward,
                split_on_char = self.split_on_char,
                expand_vocab = self.expand_vocab,
                random_case_flip = self.random_case_flip,
                document_delimiter=self.document_delimiter,
                shuffle=self.shuffle,
            )
            
            if sp =="train":
                setattr(self, k, d)
            else:
                setattr(self, k, d[0])
            

def train_model(maindataset, p_regroup, prefix):
    
    dictionary = Dictionary.load_from_file('./Models/char_mappings_v_merged')
    
    print("#Vocab:", len(dictionary.idx2item))
    
    # forward_lm
    corpus = MyTextCorpus(maindataset, dictionary, p_regroup=p_regroup, forward=True)
    language_model = LanguageModel(dictionary, True, hidden_size=128, nlayers=1)
    trainer = LanguageModelTrainer(language_model, corpus)
    epoch = 20
    model_dir = f'./Models/{prefix}_fwdLM_{p_regroup}'
        
    trainer.train(model_dir, sequence_length=280, mini_batch_size=16, max_epochs=epoch)
    
    # backward_lm
    corpus = MyTextCorpus(maindataset, dictionary, p_regroup=p_regroup, forward=False)
    language_model = LanguageModel(dictionary, False, hidden_size=128, nlayers=1)
    trainer = LanguageModelTrainer(language_model, corpus)
    
    model_dir = f'./Models/{prefix}_bkwLM_{p_regroup}'
        
    trainer.train(model_dir, sequence_length=280, mini_batch_size=16, max_epochs=epoch)
    

# maindataset = ThDatasetVISTEC("../Data/VISTEC-TP-TH-sample", name="VISTEC-sample")
maindataset = ThDatasetVISTEC("../Data/VISTEC-TP-TH-2021", name="VISTEC")

for i in range(5):
    for p in range(1, 10):
        train_model(maindataset, p_regroup=p/10.0, prefix=f"regroup/{(i+1)}")
    #     break
    # break
