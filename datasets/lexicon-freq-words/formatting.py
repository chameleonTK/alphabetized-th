import pandas as pd
from sklearn import model_selection

def reformat(d, fname):
    words = []
    for idx, row in d.iterrows():
        worden = row.en.split(",")
        wordth = row.th.split(",")
        
        for e in worden:
            for t in wordth:
                words.append((t, e, row.cnt))
        
    train, test = model_selection.train_test_split(words)

    with open(f"{fname}.txt", "w", encoding="utf-8") as fo:
        for t, e, cnt in words:
            fo.write(f"{e}\t{t}\t{cnt}\n")

    with open(f"{fname}-train.txt", "w", encoding="utf-8") as fo:
        for t, e, cnt in train:
            fo.write(f"{e}\t{t}\t{cnt}\n")

    with open(f"{fname}-test.txt", "w", encoding="utf-8") as fo:
        for t, e, cnt in test:
            fo.write(f"{e}\t{t}\t{cnt}\n")

    print("reformat", fname)
    print("#word", len(words))

data = pd.read_csv("freq_words_en_no_stopwords.tsv", sep="\t", names=["en", "th", "cnt"])
reformat(data, "en-th")

data = pd.read_csv("freq_words_th_no_stopwords.tsv", sep="\t", names=["th", "en", "cnt"])
reformat(data, "th-en")