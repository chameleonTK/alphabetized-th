from torchtext import data
from torchtext import datasets
import torchtext.vocab as vocab

import torch.optim as optim
from dan_xnli import DANXnli as DAN

import time
from tqdm import tqdm as tqdm_notebook
import torch

from gensim.models import FastText
import tokenizer as tkn

from experiment import Experiment

class XLNIExperiment(Experiment):
    def __init__(self):
        super().__init__()

    # Pre-trained
    def pretrain(self, mode, args, wv, tokenizer, full_data=False, log=False):
        args.path = './datasets/MNLI/'
        args.train_path = 'mnli_train.csv'
        args.test_path = 'mnli_dev.csv'
        if full_data:
            args.train_path = 'mnli_train_full.csv'
            args.test_path = 'mnli_dev_full.csv'

        train_iter, test_iter, TEXT, LABEL = self.init_train_test_data(args.path, args.train_path, args.test_path, wv, tokenizer, batch_size=args.batch_size)

        n_embed = len(TEXT.vocab)
        d_out = len(LABEL.vocab)
        embed_weight = TEXT.vocab.vectors
        d_embed = embed_weight.shape[1]

        model = DAN(n_embed=n_embed, d_embed=d_embed, d_hidden=256, d_out=d_out, dp=0.2, embed_weight=embed_weight)
        model = self.trainDAN(model, train_iter, test_iter, epochs=args.epochs, dev_every=args.dev_every, log=log)

        return model

    # Fine-tune
    def finetune(self, mode, args, model, wv, tokenizer, log=False):
        args.path = './datasets/XNLI/'
        args.train_path = 'xnli_dev.csv'
        args.test_path = 'xnli_test.csv'

        train_iter, test_iter, TEXT, LABEL = self.init_train_test_data(args.path, args.train_path, args.test_path, wv, tokenizer, batch_size=args.batch_size)

        n_embed = len(TEXT.vocab)
        d_out = len(LABEL.vocab)
        embed_weight = TEXT.vocab.vectors
        d_embed = embed_weight.shape[1]

        model.set_embedding(embed_weight, n_embed=n_embed, d_embed=d_embed)

        model = self.trainDAN(model, train_iter, test_iter, epochs=args.epochs, dev_every=args.dev_every, log=log)

        return model

    
if __name__ == "__main__":
    
    tokenizer = tkn.Tokenizer()
    # print(tokenizer.subwordTCCTokenize(s))
    # print(tokenizer.subwordTokenize(s))

    wv = FastText.load(f"./wv/word_th_w2v.model")
    def fnt_tokenizer(text): # create a tokenizer function
        text = tokenizer.wordTokenize(text)
        return text

    exp = XLNIExperiment()
    args = exp.get_default_arguments("demo")
    args.epochs = 1
    args.dev_every = 10

    _cnt = 0
    def _tokenizer(tok):
      def t(text):
          global _cnt
          text = tok(text)
          _cnt += 1
          if _cnt <=5:
            print(text)
          return text
      return t
    
    
    _cnt = 0
    model = exp.pretrain("word_en", args, wv.wv, _tokenizer(tokenizer.wordEnTokenize))

    _cnt = 0
    model = exp.finetune("word_en", args, model, wv.wv, _tokenizer(tokenizer.wordTokenize))

    model.save_model("./models/demo.pt")
    newmodel = DAN.load_model("./models/demo.pt")

    
    # print(newmodel, newmodel.fc2.weight, model.fc2.weight)
