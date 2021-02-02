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
        "../oscar/word_th_tcc/oscar_word_th_tcc1.txt",
        "../oscar/word_th_tcc/oscar_word_th_tcc2.txt",
        "../oscar/word_th_tcc/oscar_word_th_tcc3.txt",
        "../oscar/word_th_tcc/oscar_word_th_tcc4.txt",
        "../oscar/word_th_tcc/oscar_word_th_tcc5.txt",
        #"../oscar/word_th_tcc/oscar_word_th_tcc6.txt",
        #"../oscar/word_th_tcc/oscar_word_th_tcc7.txt",
        #"../oscar/word_th_tcc/oscar_word_th_tcc8.txt",
        #"../oscar/word_th_tcc/oscar_word_th_tcc9.txt",
    ]
    
    for i in [1, 2, 3, 5]:

        print("Loading text ",i)
        tmp_filename = "_tmp_merged_text_tcc.txt"
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

        

        print("Loaded text")
        t = time()
        model = fasttext.train_unsupervised(tmp_filename, ws=5, minCount=5, minn=2, maxn=5, dim=300)
        model.save_model(f"wv/oscar_word_th_tcc_{i}M.bin")

        print("==============")
        print("Train:", f"oscar_word_th_tcc_{i}M")
        print("Number of sentences", nsent)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        print("vocab size:", len(model.words))
