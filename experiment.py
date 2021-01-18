import os
import time
import glob

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import os
import sys
from argparse import ArgumentParser

from torchtext import data
from torchtext import datasets
import torchtext.vocab as vocab

from tqdm import tqdm_notebook

class Experiment:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss() 
        self.args = self.get_default_arguments("")
    
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def get_default_arguments(self, name):
        parser = ArgumentParser(description='DAN')
        parser.add_argument('mode', type=str, help = 'tokenizing mode ')
        parser.add_argument('--epochs', type=int, default=50, help = 'epochs')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--d_embed', type=int, default=100)
        parser.add_argument('--lr', type=float, default=.001)
        parser.add_argument('--dev_every', type=int, default=100)
        parser.add_argument('--dp_ratio', type=int, default=0.2)
        parser.add_argument('--gpu', type=int, default=0)

        try:
            args = parser.parse_args([f"model={name}"])
        except:
            parser.print_help()
            raise Exception("Cannot parse arguments")

        return args

    def evaluate(self, loader, model):
        model.eval()
        loader.sort = False
        loader.sort_within_batch = False
        loader.init_epoch()

        criterion = self.criterion
        # calculate accuracy on validation set
        n_correct, n = 0, 0
        losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                answer = model(batch)
                n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
                n += answer.shape[0]
                loss = criterion(answer, batch.label)
                losses.append(loss.data.cpu().numpy())
        acc = 100. * n_correct / n
        loss = np.mean(losses)

        return acc, loss

    def init_w2v_from_pretrained(self, wv, vocabs):
        W2V_SIZE = wv.vector_size
        # W2V_WINDOW = wv.window
        W2V_WINDOW = 5
        # W2V_MIN_COUNT = wv.min_count
        W2V_MIN_COUNT = 5

        words = wv.vocab.keys()
        vocab_size = len(words)

        # ref: https://medium.com/@rohit_agrawal/using-fine-tuned-gensim-word2vec-embeddings-with-torchtext-and-pytorch-17eea2883cd
        word2vec_vectors = []
        for token, idx in tqdm_notebook(vocabs):
            if token in wv.vocab.keys():
                word2vec_vectors.append(torch.FloatTensor(wv[token].copy()))
            else:
                word2vec_vectors.append(torch.zeros(W2V_SIZE))
        return word2vec_vectors

    def init_train_test_data(self, path, train_path, test_path, wv, tokenizer, batch_size=32):
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        LABEL = data.Field(sequential=False, use_vocab=False)

        train, test = data.TabularDataset.splits(
            path=path, 
            train=train_path, test=test_path, format='csv',
            skip_header=True,
            fields=[('sentence1', TEXT), ('sentence2', TEXT), ('label', LABEL)])
        
        # W2V_MIN_COUNT = wv.min_count
        W2V_MIN_COUNT = 5
        W2V_SIZE = wv.vector_size
        TEXT.build_vocab(train, test, min_freq=W2V_MIN_COUNT, )
        vocabs = TEXT.vocab.stoi.items()
        word2vec_vectors = self.init_w2v_from_pretrained(wv, vocabs)
        TEXT.vocab.set_vectors(TEXT.vocab.stoi, word2vec_vectors, W2V_SIZE)

        LABEL.build_vocab(train)

        train_iter, test_iter = data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=self.device)

        return train_iter, test_iter, TEXT, LABEL


    def trainDAN(self, model, train_iter, test_iter, lr=.001, epochs=10, save_path=None, dev_every=10, log=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
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

        for epoch in range(epochs):
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
                    epoch, (batch_idx + 1), len(train_iter), loss.item(),
                            train_loss / (iterations - last_val_iter)), end='')
                
                if log:
                    wandb.log({'iter': iterations, 'epoch': epoch, 'train_loss': loss.item()})                
                    
                if iterations > 0 and iterations % dev_every == 0:
                    acc, val_loss = self.evaluate(test_iter, model)
                    _save_ckp = '*'
                    if acc > best_acc:
                        best_acc = acc
                        if save_path is not None:
                            torch.save(model.state_dict(), save_path)

                    print(
                        ' {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} |'.format(
                            val_loss, acc, best_acc, (time.time() - start) / 60,
                            _save_ckp))
        
                    if log:
                        wandb.log({'iter': iterations, 'epoch': epoch, 'val_loss': val_loss, 'acc': acc})

                    train_loss = 0
                    last_val_iter = iterations
      
        return model