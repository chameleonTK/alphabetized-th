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


from gensim.models import KeyedVectors
import sys

if __name__ == "__main__":
    
    print("XNLI Evaluation")
    if len(sys.argv) > 1:
        wvname = sys.argv[1]
        print("Loading EN wv", wvname)
        wven = KeyedVectors.load_word2vec_format(f"{wvname}/vectors-en.txt")

        thcol = sys.argv[2] if len(sys.argv) > 1 else "th"
        print("Loading TH wv", wvname, thcol)
        wvth = KeyedVectors.load_word2vec_format(f"{wvname}/vectors-{thcol}.txt")
        
    else:
        print("Please specify wordvector location")
        sys.exit(0)

    tokenizer = tkn.Tokenizer()
    

    exp = XLNIExperiment()
    args = exp.get_default_arguments("XNLI")
    args.epochs = 5
    args.dev_every = 10
    model = exp.pretrain("word_en_th", args, wven, tokenizer.wordEnTokenize)

    if thcol=="th":
        thtokenizer = tokenizer.wordTokenize
    else:
        thtokenizer = tokenizer.wordTCCTokenize

    model = exp.finetune("word_en_th", args, model, wvth, thtokenizer)

    saved_path = sys.argv[3] if len(sys.argv) > 2 else "./models/demo.pt"
    model.save_model(saved_path)
    # newmodel = DAN.load_model("./models/demo.pt")

    
    # print(newmodel, newmodel.fc2.weight, model.fc2.weight)
