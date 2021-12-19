#import torch
import tokenizer as tkn
from util import Util

# from gensim.models import Word2Vec
# from gensim.models import FastText
#import multiprocessing


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
from tqdm import tqdm
import sys
from util import Util
if __name__ == "__main__":
    #if torch.cuda.is_available():    
    #    device = torch.device("cuda")
    #else:
    #    device = torch.device("cpu")

    util = Util()
    base_dir = "/import/cogsci/tk/oscar/"
    filenames = [
        "raw_th/th_part_1.txt",
        "raw_th/th_part_2.txt",
        "raw_th/th_part_3.txt",
        "raw_th/th_part_4.txt",
        "raw_th/th_part_5.txt",
        #"/homes/pn004/raw_th/th_part_6.txt",
        #"/homes/pn004/raw_th/th_part_7.txt",
        #"/homes/pn004/raw_th/th_part_8.txt",
        #"/homes/pn004/raw_th/th_part_9.txt",
    ]
    print(sys.argv)
    thcol = sys.argv[1] if len(sys.argv) > 1 else "th"

    print("MODE", thcol)

    tokenizer = tkn.Tokenizer()
    util = Util()
    for i in [5]:

        print("Loading text ",i)
        tmp_filename = f"_tmp_merged_text_{thcol}.txt"
        fo = open(tmp_filename, "w")
        nsent = 0
        done = False
        pbar = tqdm(total=1e6*i)
        for fname in filenames:
            with open(base_dir+fname) as fin:
                for line in fin:
                    #print(line)
                    #print(tokenizer.subwordTokenize(line))
                    #print(tokenizer.subwordTCCTokenize(line))
                    #break

                    if thcol!="th":
                        line = util.tcc_encode(line)
                    line = line.replace(" ", "_")
                    line = tokenizer.characterTokenize(line)
                    line = " ".join(line)
                    fo.write(line+"\n")
                    nsent += 1
                    pbar.update(1)

                    if nsent > (1e6*i):
                        done = True
                        break
            if done:
                break
        fo.close()

        
        print("Loaded text")
        t = time()
        model = fasttext.train_unsupervised(tmp_filename, ws=5, minCount=5, minn=2, maxn=5, dim=300)
        model.save_model(f"wv/oscar_char_{thcol}_{i}M.bin")

        print("==============")
        print("Train:", f"oscar_word_{thcol}_{i}M")
        print("Number of sentences", nsent)
        print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
        print("vocab size:", len(model.words))
