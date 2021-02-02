import pandas as pd
import numpy as np
import fasttext

import logging, sys
import scipy.stats
from six import iteritems

def fetch_TWS65():
    """
    added by Gerhard Wohlgenannt, (gwohlg@corp.ifmo.ru, wohlg@ai.wu.ac.at)
    Get the TWS65 dataset for Thai language
    The dataset is originally from this thesis: https://e-space.mmu.ac.uk/336070/
    by Osathanunkul, Khukrit (2014)

    The dataset is in Thai language (!) for the evaluation of Thai embedding models

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    References
    ----------
    Osathanunkul, Khukrit (2014) Semantic similarity framework for Thai conversational agents. Doctoral thesis (PhD), Manchester Metropolitan University.

    """

    return load_csv("datasets/word_similarity/tws65.csv")


def fetch_thai_wordsim353():
    """
    added by Gerhard Wohlgenannt, (gwohlg@corp.ifmo.ru, wohlg@ai.wu.ac.at), 2019
    Get the WordSim-353 dataset for Thai language
    
    The dataset is in Thai language (!) for the evaluation of Thai embedding models

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    """
    return load_csv("datasets/word_similarity/thai-wordsim353-v2.csv")


def fetch_thai_semeval2017_task2():
    """
    added by Gerhard Wohlgenannt, (gwohlg@corp.ifmo.ru, wohlg@ai.wu.ac.at), 2019
    Get the SemEval2017-Task2 dataset for Thai language
    
    The dataset is in Thai language (!) for the evaluation of Thai embedding models

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    """
    return load_csv("datasets/word_similarity/thaiSemEval-500-v2.csv")


def fetch_thai_simlex999():
    """
    added by Gerhard Wohlgenannt, (gwohlg@corp.ifmo.ru, wohlg@ai.wu.ac.at), 2019
    Get the SemEval2017-Task2 dataset for Thai language
    
    The dataset is in Thai language (!) for the evaluation of Thai embedding models

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    """
    
    return load_csv("datasets/word_similarity/thaiSimLex-999-v2.csv")

def load_csv(path):
    d = pd.read_csv(path, names=["w1", "w2", "score"])
    return {
        "X": d[["w1", "w2"]].values,
        "y": d["score"].values
    }

def evaluate_similarity(wv, X, y, preprocess=None, output=None):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs
    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.
    X: array, shape: (n_samples, 2)
      Word pairs
    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """

    missing_words, found_words, oov_vecs_created, index = 0, 0, 0, 0
    word_pair_oov_indices = []
    info_oov_words = {}
    info_created_words = {}


    words = wv.get_words()
    # For all words in the datasets, check if the are OOV? 
    # Indices of word-pairs with a OOV word are stored in word_pair_oov_indices
    
    def do_nothing(x):
      return x
      
    if preprocess is None:
      print("NONE")
      preprocess = do_nothing

    nwords = 0
    for query in X:
        for w in query:
            w = preprocess(w)
            if w not in words:
                missing_words += 1
                word_pair_oov_indices.append(index)
            else:
                # print("Found Word:", query_word)
                found_words += 1
            nwords += 1
        index += 1

    # print(f"Missing Word: {missing_words} words ({missing_words*100/nwords:.2f}%)")
    

    # The original code; for all OOV; it will be replaced by average vector
    # mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    # A = np.vstack(w.get(word, mean_vector) for word in X[:, 0])
    # B = np.vstack(w.get(word, mean_vector) for word in X[:, 1])
    # scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])


    A = np.vstack(wv.get_word_vector(preprocess(word)) for word in X[:, 0])
    B = np.vstack(wv.get_word_vector(preprocess(word)) for word in X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])


    if output is not None:
        d = pd.DataFrame({
            "w1": X[:, 0],
            "w2": X[:, 1],
            "predict": scores,
            "actual": y
        })
        d.to_csv(output+".csv")
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


import json
from util import Util

if __name__ == "__main__":
    print("Word Similarity Evaluation")
    if len(sys.argv) > 1:
        wvname = sys.argv[1]
        wv = fasttext.load_model(wvname)
        print("Loading ", wvname)
    else:
        print("Please specify wordvector.bin")
        sys.exit(0)

    tasks = {
        "TH-WS353": fetch_thai_wordsim353(),
        "TH-SemEval2017T2": fetch_thai_semeval2017_task2(),
        "TH-SimLex999": fetch_thai_simlex999(),
        "TWS65": fetch_TWS65()
    }

    
    df = []
    util = Util()

    # Print sample data
    for name, data in iteritems(tasks):
        print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".format(name, data["X"][0][0], data["X"][0][1], data["y"][0]))

    # Calculate results using helper function for the various word similarity datasets
    for name, data in iteritems(tasks):
        print("NEW TASK:", name)
        output = sys.argv[2] if len(sys.argv) > 2 else None
        result = evaluate_similarity(wv, data["X"], data["y"], output=output+name, preprocess=util.tcc_encode)

        # hm = scipy.stats.hmean([result['spearmanr'], result['pearsonr']])
        perc_oov_words = 100 * (result['num_missing_words'] / (result['num_found_words'] + float(result['num_missing_words'])))
    
        # Spearman: evaluate a monotonic relationship between two variables based on the ranked values for each variable rather than the raw data.
        # Pearson : measures the linear correlation between two variables X and Y        # Pearson : measures the linear correlation between two variables X and Y
        print("""Dataset {:17}: \nSpearman/Pearson/HarmMean: {:4.3f} {:4.3f} \nOOV percentage: {:4.3f}""".format(
            name, 
            round(result['spearmanr'],3), 
            round(result['pearsonr'],3),
            perc_oov_words
        ))
        print("\n\n")
        result["name"] = name 
        df.append(result)

    print(json.dumps(df))
