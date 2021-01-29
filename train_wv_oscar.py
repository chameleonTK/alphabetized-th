import torch
import tokenizer as tkn
from util import Util

# from gensim.models import Word2Vec
# from gensim.models import FastText
import multiprocessing


from time import time
import fasttext

# def load_sents(path):
#     # Load pre-segmented corpus

#     sents = []
#     with open(path, encoding="utf-8") as fin:
#         for line in fin:
#             line = line.strip()
#             line = line.split(" ")
#             line = list(filter(None, line)) # filter empty string
#             sents.append(line)

#             if len(sents) >= 1e3:
#                 break
#     return sents

if __name__ == "__main__":
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    util = Util()

    filenames = [
        "../oscar/word/oscar_word_th1.txt",
        "../oscar/word/oscar_word_th2.txt",
        "../oscar/word/oscar_word_th3.txt",
        "../oscar/word/oscar_word_th4.txt",
        "../oscar/word/oscar_word_th5.txt",
        "../oscar/word/oscar_word_th6.txt",
        "../oscar/word/oscar_word_th7.txt",
        "../oscar/word/oscar_word_th8.txt",
        "../oscar/word/oscar_word_th9.txt",
    ]

    
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

        tmp_filename = "_tmp_merged_text.txt"
        fo = open(tmp_filename, "w")
        nsent = 0
        done = False
        for fname in filenames:
            with open(fname) as fin:
                for line in fin:
                    fo.write(line)
                    nsent += 1

                    if nsent > (1e6*i):
                        done = True
                        break
            if done:
                break
        fo.close()

        

        
        t = time()
        model = fasttext.train_unsupervised(tmp_filename, ws=5, minCount=5, minn=2, maxn=5, dim=300)
        model.save_model(f"wv/oscar_word_th_{i}M.bin")

        print("==============")
        print("Train:", f"oscar_word_th_{i}M")
        print("Number of sentences", nsent)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        print("vocab size:", len(model.words))