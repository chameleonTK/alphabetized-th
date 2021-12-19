from torchtext import data
from torchtext import datasets
import torchtext.vocab as vocab

import torch.optim as optim
from dan import DAN

import time
from tqdm import tqdm as tqdm_notebook
import torch

import tokenizer as tkn

from experiment import Experiment
import wandb
import sys
import fasttext
import numpy as np

class BasicExperiment(Experiment):
    def __init__(self):
        super().__init__()

    def train(self, mode, args, tokenizer, wv, log=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)

        # vector_size = model.wv.vector_size
        # print(model.wv.vectors_vocab.shape)

        train, test = data.TabularDataset.splits(
            path='./datasets/Wisesight-sentiment/', 
            train='wisesight_train.csv', test='wisesight_test.csv', format='csv',
            skip_header=True,
            fields=[('norm_text', TEXT), ('label', LABEL)])
        
        W2V_SIZE = len(wv[wv.words[0]]),
        W2V_MIN_COUNT = 5

        words = set(wv.words)
        vocab_size = len(words)

        TEXT.build_vocab(train, test, min_freq=W2V_MIN_COUNT, )

        # ref: https://medium.com/@rohit_agrawal/using-fine-tuned-gensim-word2vec-embeddings-with-torchtext-and-pytorch-17eea2883cd
        word2vec_vectors = []
        for token, idx in tqdm_notebook(TEXT.vocab.stoi.items()):
            if token in words:
                word2vec_vectors.append(torch.FloatTensor(wv[token].copy()))
            else:
                word2vec_vectors.append(torch.zeros(W2V_SIZE))
                
        TEXT.vocab.set_vectors(TEXT.vocab.stoi, word2vec_vectors, W2V_SIZE[0])

        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits(
            (train, test), batch_size=args.batch_size, device=device)

        n_embed = len(TEXT.vocab)
        d_out = len(LABEL.vocab)

        model = DAN(n_embed=n_embed, d_embed=args.d_embed, d_hidden=256, d_out=d_out, dp=0.2, embed_weight=TEXT.vocab.vectors)
        model.to(device)

        opt = optim.Adam(model.parameters(), lr=args.lr)

        acc, val_loss = self.evaluate(test_iter, model)

        best_acc = acc

        print(
            'epoch |   %        |  loss  |  avg   |val loss|   acc   |  best  | time | save |')
        print(
            'val   |            |        |        | {:.4f} | {:.4f} | {:.4f} |      |      |'.format(
                val_loss, acc, best_acc))

        iterations = 0
        last_val_iter = 0
        train_loss = 0
        start = time.time()

        for epoch in range(args.epochs):
            train_iter.init_epoch()
            n_correct, n_total, train_loss = 0, 0, 0
            last_val_iter = 0
            for batch_idx, batch in enumerate(train_iter):
                # switch model to training mode, clear gradient accumulators
                model.train();
                opt.zero_grad()

                iterations += 1

                # forward pass
                answer = model(batch)
                loss = self.criterion(answer, batch.label)

                loss.backward();
                opt.step()

                train_loss += loss.item()
                print('\r {:4d} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                    epoch, args.batch_size * (batch_idx + 1), len(train), loss.item(),
                            train_loss / (iterations - last_val_iter)), end='')

                if log:
                    wandb.log({'iter': iterations, 'epoch': epoch, 'train_loss': loss.item()})                
                if iterations > 0 and iterations % args.dev_every == 0:
                    acc, val_loss = self.evaluate(test_iter, model)
                    _save_ckp = '*'
                    if acc > best_acc:
                        best_acc = acc
                        # torch.save(model.state_dict(), args.save_path)

                    print(
                        ' {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} |'.format(
                            val_loss, acc, best_acc, (time.time() - start) / 60,
                            _save_ckp))
                    
                    if log:
                        wandb.log({'iter': iterations, 'epoch': epoch, 'val_loss': val_loss, 'acc': acc})

                    train_loss = 0
                    last_val_iter = iterations
        return model, best_acc


if __name__ == "__main__":
    
    tokenizer = tkn.Tokenizer()
    # print(tokenizer.subwordTCCTokenize(s))
    # print(tokenizer.subwordTokenize(s))

    print("Arguments", sys.argv)
    wvpath = sys.argv[1]
    
    mode = sys.argv[2] if len(sys.argv) > 1 else "word"
    if mode=="word":
        fnt_tokenizer = tokenizer.wordTokenize
    elif mode=="wordtcc":
        fnt_tokenizer = tokenizer.wordTCCTokenize
    elif mode=="subword":
        fnt_tokenizer = tokenizer.subwordTokenize
    elif mode=="subwordtcc":
        fnt_tokenizer = tokenizer.subwordTCCTokenize
    elif mode=="char":
        fnt_tokenizer = tokenizer.characterTokenize
    elif mode=="chartcc":
        fnt_tokenizer = tokenizer.characterTCCTokenize
    else:
        raise Exception("Unknown mode")

    wv = fasttext.load_model(wvpath)

    acc = []
    for i in range(5):
        exp = BasicExperiment()
        args = exp.get_default_arguments("demo")
        args.epochs = 10
        args.d_embed = 300                
        model, best_acc = exp.train("demo", args, fnt_tokenizer, wv)

        acc.append(best_acc)
    print(np.mean(acc), np.std(acc))
    # model.save_model("./models/demo.pt")

    # newmodel = DAN.load_model("./models/demo.pt")
    # print(newmodel, newmodel.fc2.weight, model.fc2.weight)
