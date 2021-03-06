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

import sys
if __name__ == "__main__":
    if torch.cuda.is_available():    
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    util = Util()

    filenames = [
        "/homes/pn004/raw_th/th_part_1.txt",
        "/homes/pn004/raw_th/th_part_2.txt",
        "/homes/pn004/raw_th/th_part_3.txt",
        "/homes/pn004/raw_th/th_part_4.txt",
        "/homes/pn004/raw_th/th_part_5.txt",
        #"/homes/pn004/raw_th/th_part_6.txt",
        #"/homes/pn004/raw_th/th_part_7.txt",
        #"/homes/pn004/raw_th/th_part_8.txt",
        #"/homes/pn004/raw_th/th_part_9.txt",
    ]

    thcol = sys.argv[2] if len(sys.argv) > 1 else "th"

    print("MODE", thcol)

    tokenizer = tkn.Tokenizer()
    for i in [5, 10]:

        print("Loading text ",i)
        tmp_filename = f"_tmp_merged_text_{thcol}.txt"
        fo = open(tmp_filename, "w")
        nsent = 0
        done = False
        for fname in filenames:
            with open(fname) as fin:
                for line in fin:
                    print(line)
                    print(tokenizer.subwordTokenize(line))
                    print(tokenizer.subwordTCCTokenize(line))
                    break

                    if thcol=="th":
                        line = tokenizer.subwordTokenize(line)
                    else:
                        line = tokenizer.subwordTCCTokenize(line)
                    fo.write(line)
                    nsent += 1

                    if nsent > (1e6*i):
                        done = True
                        break
            if done:
                break
        fo.close()

        
        break
        print("Loaded text")
        t = time()
        model = fasttext.train_unsupervised(tmp_filename, ws=5, minCount=5, minn=2, maxn=5, dim=300)
        model.save_model(f"wv/oscar_subword_{thcol}_{i}M.bin")

        print("==============")
        print("Train:", f"oscar_word_{thcol}_{i}M")
        print("Number of sentences", nsent)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        print("vocab size:", len(model.words))
