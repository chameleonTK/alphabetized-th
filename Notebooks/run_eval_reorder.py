import sys
sys.path.append('../Libs/word-embeddings-benchmarks/')

from flair.data import Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus


import logging, sys
import scipy.stats
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.datasets.similarity import fetch_TWS65, fetch_thai_wordsim353, fetch_thai_semeval2017_task2, fetch_thai_simlex999
from web.embeddings import load_embedding
# from web.evaluate import evaluate_similarity
import sklearn
import numpy as np
from scipy import spatial

from util import *

import pickle
import random

def transform_reorder(sent, p, window=4):
    random.seed(42)
    tokens = list(sent)
    
    for i, t in enumerate(tokens):
        span = tokens[i:i+window]
        if random.random() < p:
            a = random.randint(0, window-1)
            b = random.randint(0, window-1)
            try:
                _tmp = tokens[i+a]
                tokens[i+a] = tokens[i+b]
                tokens[i+b] = _tmp
            except:
                continue
                
    return tokens



import pickle

def evaluate_similarity(wv, X, y, preprocess=None):
    
    missing_words, found_words, oov_vecs_created, index = 0, 0, 0, 0
    word_pair_oov_indices = []
    info_oov_words = {}
    info_created_words = {}

    ## For all words in the datasets, check if the are OOV? 
    ## Indices of word-pairs with a OOV word are stored in word_pair_oov_indices
    
    nwords = 0
    for query in X:
        for query_word in query:
            found_words += 1
            nwords += 1
        index += 1

    # print(f"Missing Word: {missing_words} words ({missing_words*100/nwords:.2f}%)")
    

    # The original code; for all OOV; it will be replaced by average vector
    # mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    # A = np.vstack(w.get(word, mean_vector) for word in X[:, 0])
    # B = np.vstack(w.get(word, mean_vector) for word in X[:, 1])
    # scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    
    scores = []
    for w in X:
        vecA = wv.get_vector(w[0])
        vecB = wv.get_vector(w[1])
        s = 1 - spatial.distance.cosine(vecA, vecB)
        scores.append(s)
        
#     A = np.vstack(w[preprocess(word)] for word in )
#     B = np.vstack(w[preprocess(word)] for word in X[:, 1])
#     scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])


    # wohlg: original version only returned Spearman 
    # wohlg: we added Pearson and other information 
    result = {
        'spearmanr': scipy.stats.spearmanr(scores, y).correlation,
        'pearsonr':  scipy.stats.pearsonr(scores, y)[0],
        'num_oov_word_pairs': len(word_pair_oov_indices),
        'num_found_words': found_words,
        'num_missing_words': missing_words,
        "num_word_pairs": nwords
    }

    return result

tasks = {
    "TH-WS353": fetch_thai_wordsim353(),
    "TH-SemEval2017T2": fetch_thai_semeval2017_task2(),
    "TH-SimLex999": fetch_thai_simlex999(),
    "TWS65": fetch_TWS65()
}

# Print sample data
for name, data in iteritems(tasks):
    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data.X[0][0], data.X[0][1], data.y[0]))

def eval_word_sim(wv, expname="", verbose=True):
    # Calculate results using helper function for the various word similarity datasets
    results = {}
    for name, data in iteritems(tasks):
        result = evaluate_similarity(wv, data.X, data.y)

    #     hm = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
        perc_oov_words = 100 * (result['num_missing_words'] / (result['num_found_words'] + float(result['num_missing_words'])))

        # Spearman: evaluate a monotonic relationship between two variables based on the ranked values for each variable rather than the raw data.
        # Pearson : measures the linear correlation between two variables X and Y
        if verbose:
            print(f"{expname}:Dataset {name}: Spearman: {result['spearmanr']:4.3f}")
        results[name] = result['spearmanr']
    return results

from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence

class WordVector:
    def __init__(self, p_reorder, fwd=True, bkw=True, model_dir_prefix="", model_dir_suffix=""):
        embs = []
        if fwd:
            flair_embedding_forward = FlairEmbeddings(f"./Models/{model_dir_prefix}fwdLM{model_dir_suffix}/best-lm.pt")
            embs.append(flair_embedding_forward)
        
        if bkw:
            flair_embedding_backward = FlairEmbeddings(f"./Models/{model_dir_prefix}bkwLM{model_dir_suffix}/best-lm.pt")
            embs.append(flair_embedding_backward)
        
        self.p_reorder = p_reorder
        self.stacked_embeddings = StackedEmbeddings(embs)

        
    def get_vector(self, word):
        word = "".join(transform_reorder(word, p=self.p_reorder))
            
        sentence = Sentence([word])
        self.stacked_embeddings.embed(sentence)

        for token in sentence:
            return token.embedding.cpu()


for i in range(5):
    for p in range(1, 10):
        p_reorder=p/10.0
        print(f"Reorder {i+1}")
        wv = WordVector(p_reorder=p_reorder, model_dir_prefix=f"reorder/{i+1}_", model_dir_suffix=f"_{p_reorder}")
        wordsim = eval_word_sim(wv, expname=f"Reorder {i+1}:{p_reorder}:")