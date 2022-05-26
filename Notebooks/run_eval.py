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
additionTokens = {}
with open(f'./Models/alphs_addition_tokens', 'rb') as f:
    additionTokens = pickle.load(f)
    

def custom_alphabetize(sent):
    tokens = alphabetize(sent)
    
    newtokens = []
    for t in tokens:
        if len(t) > 1:
            newtokens.append(additionTokens[t])
        else:
            newtokens.append(t)
    
    return newtokens


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

def eval_word_sim(wv, verbose=True):
    # Calculate results using helper function for the various word similarity datasets
    results = {}
    for name, data in iteritems(tasks):
        result = evaluate_similarity(wv, data.X, data.y)

    #     hm = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
        perc_oov_words = 100 * (result['num_missing_words'] / (result['num_found_words'] + float(result['num_missing_words'])))

        # Spearman: evaluate a monotonic relationship between two variables based on the ranked values for each variable rather than the raw data.
        # Pearson : measures the linear correlation between two variables X and Y
        if verbose:
            print(f"Dataset {name}: Spearman: {result['spearmanr']:4.3f}")
        results[name] = result['spearmanr']
    return results

from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence

class WordVector:
    def __init__(self, alphabetized, fwd=True, bkw=True):
        model_dir_suffix = ""
        if alphabetized:
            model_dir_suffix = "_alph"
        
        embs = []
        if fwd:
            flair_embedding_forward = FlairEmbeddings(f"./Models/fwdLM{model_dir_suffix}/best-lm.pt")
            embs.append(flair_embedding_forward)
        
        if bkw:
            flair_embedding_backward = FlairEmbeddings(f"./Models/bkwLM{model_dir_suffix}/best-lm.pt")
            embs.append(flair_embedding_backward)
        
        self.alphabetized = alphabetized
        self.stacked_embeddings = StackedEmbeddings(embs)

        
    def get_vector(self, word):
        if self.alphabetized:
#             print(word, "".join(custom_alphabetize(word)))
            word = "".join(custom_alphabetize(word))
            
        sentence = Sentence([word])
        self.stacked_embeddings.embed(sentence)

        for token in sentence:
            return token.embedding.cpu()
print("\n=======")
print("Alphabetized")
wv = WordVector(alphabetized=True)
wordsim = eval_word_sim(wv)

print("\n=======")
print("Baseline")
wv = WordVector(alphabetized=False)
wordsim = eval_word_sim(wv)