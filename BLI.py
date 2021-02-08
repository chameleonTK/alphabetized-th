# from collections import defaultdict 
# from pythainlp import corpus
# from nltk.corpus import stopwords as stopw_en
# nltk.download('stopwords')
import numpy as np
from tqdm import tqdm
# from util import Util

import sys
import io
import torch

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    with tqdm(total=n) as pbar:
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = [float(n) for n in tokens[1:]]
            pbar.update(1)
    return data


def load_dictionary(path, word2id1, word2id2):
    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0
    with io.open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            line = line.lower()
            parts = line.rstrip().split()
            if len(parts) < 2:
                print("Could not parse line %s (%i)", line, index)
                continue
            word1 = parts[0]
            word2 = parts[1]

            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)
    print("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]
    return dico

if __name__ == "__main__":
    print("BLI Evaluation")
    # stopwords_th = corpus.thai_stopwords()
    # stopwords_en = set(stopw_en.words())
    if len(sys.argv) > 1:
        aligned_model = sys.argv[1]

        thcol = sys.argv[2] if len(sys.argv) > 2 else "th"
        wven = load_vectors(f"{aligned_model}vectors-en.txt")
        wvth = load_vectors(f"{aligned_model}vectors-{thcol}.txt")

        print("Loading ", aligned_model, thcol)
    else:
        print("Please specify wordvector")
        sys.exit(0)

    word2id1 = {w:idx for idx, w in enumerate(wven.keys())}
    word2id2 = {w:idx for idx, w in enumerate(wvth.keys())}

    lexicon_path = [
        f"datasets/lexicon-en-th/en-{thcol}.5000-6500.txt",
        # "datasets/lexicon-freq-words/en-th.txt"
    ]
    
    for path in lexicon_path:
        print("Path", path)
        dico = load_dictionary(path, word2id1, word2id2)
        dico = dico
        
        emb1 = torch.Tensor([wven[k] for k in wven])
        emb1_norm = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)

        emb2 = torch.Tensor([wvth[k] for k in wvth])
        emb2_norm = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

        w0 = dico[:, 0]
        query = emb1[w0]

        scores = query.mm(emb2_norm.transpose(0, 1))
        inds = scores.argsort(axis=1, descending=True)
        w1 = dico[:, 1]

        w1 = w1[:, None].expand_as(inds)
        matched = (inds==w1)+0

        p = torch.argmax(matched, axis=1)+1
        mrr = torch.sum(1/p)/len(p)

        p1 = (p==1).sum()/len(p)
        p5 = (p<=5).sum()/len(p)
        p10 = (p<=10).sum()/len(p)

        print("MRR", mrr)
        print("P@1", p1)
        print("P@5", p5)
        print("P@10", p10)
