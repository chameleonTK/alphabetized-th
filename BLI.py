from collections import defaultdict 
# from pythainlp import corpus
# from nltk.corpus import stopwords as stopw_en
# nltk.download('stopwords')
import numpy as np, pandas as pd
from tqdm.notebook import tqdm
from util import Util
from gensim.models import KeyedVectors

def load_dictionary():
    dictionary = defaultdict(lambda: set())
    with open("datasets/Yaitron/yaitron_par.tsv", encoding="utf-8") as fin:
        for line in fin:
            en, th = line.strip().split("\t")
            
            en = en.lower()
            dictionary[en].add(th)
            dictionary[th].add(en)

def most_similar_index(word_src, word_trgs, src_embs, trg_embs, trg_vocabs, verbose=False):
    if word_src not in src_embs:
        if verbose:
            print("Word src not found in vocabulary: " + word_src)
        return None

    filtered_word_trg = []
    for w in word_trgs:
        if w in trg_embs:
            filtered_word_trg.append(w)
    
    if len(filtered_word_trg)==0:
        if verbose:
            print("Word trg not found in vocabulary: " + str(word_trgs))
        return None
    
    word_src_emb = src_embs[word_src]
    sims = trg_embs.distances(word_src_emb)
    
    inds = np.argsort(sims)
    if verbose:
        for idx in inds[0:5]:
            print(trg_vocabs[idx])

    idxs = []
    for w in filtered_word_trg:
        trg_ind = np.where(trg_vocabs == w)[0][0]
        idx = np.where(inds == trg_ind)[0][0] + 1
        idxs.append(idx)
    return min(idxs)


def eval_BLI(df, lang_keys, wv_src, wv_trg):
  src, trg = lang_keys
  vocabs = np.array(list(wv_trg.vocab.keys()))
  positions = []
  for index, row in tqdm(df.iterrows(), total=len(df)):
    ws = row[src].split(",")
    wt = row[trg].split(",")
    
    idx = most_similar_index(ws[0], wt, wv_src, wv_trg, vocabs)
    positions.append(idx)
  
  
  positions = list(filter(lambda x: x is not None, positions))
  p1 = len([p for p in positions if p == 1]) / len(positions)
  p5 = len([p for p in positions if p <= 5]) / len(positions)
  p10 = len([p for p in positions if p <= 10]) / len(positions)
  mrr = sum([1.0/p for p in positions]) / len(positions)


  return {
      "not_none": len(positions),
      "P1": p1,
      "P5": p5,
      "P10": p10,
      "MRR": mrr
  }

import sys
if __name__ == "__main__":
    print("BLI Evaluation")
    # stopwords_th = corpus.thai_stopwords()
    # stopwords_en = set(stopw_en.words())
    if len(sys.argv) > 1:
        aligned_model = sys.argv[1]

        thcol = sys.argv[2] if len(sys.argv) > 2 else "th"
        wven = KeyedVectors.load_word2vec_format(f"{aligned_model}/vectors-en-norm.txt")
        wvth = KeyedVectors.load_word2vec_format(f"{aligned_model}/vectors-{thcol}-norm.txt")

        print("Loading ", aligned_model, thcol)
    else:
        print("Please specify wordvector")
        sys.exit(0)

    util = Util()
    dictionary = load_dictionary()

    dth = pd.read_csv("datasets/freq_words/freq_words_th_no_stopwords.tsv", sep="\t", names=["th", "en", "count"])
    den = pd.read_csv("datasets/freq_words/freq_words_en_no_stopwords.tsv", sep="\t", names=["en", "th", "count"])
    den["thtcc"] = den.th.apply(util.tcc_encode)
    dth["thtcc"] = dth.th.apply(util.tcc_encode)

    
    
    
    topk = 1000
    p1 = eval_BLI(dth.head(topk), (thcol, "en"), wven, wvth)
    p2 = eval_BLI(den.head(topk), ("en", thcol), wven, wvth)
    print(p1)
    print(p2)
    