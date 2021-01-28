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

    # util = Util()

    # k = "word_th"
    # t = time()
    # model = fasttext.train_unsupervised('datasets/Oscar/df_samples.txt', ws=5, minCount=5, minn=2, maxn=5, dim=300)
    # model.save_model("wv/newwv.bin")

    # print("==============")
    # print("Train:", k)
    # print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    # print("vocab size:", len(model.words))
    
    