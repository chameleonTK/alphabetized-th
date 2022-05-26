# Thai Misspelling Correction

## Version
* v1.0 
* v2.0 reimplement
* v2.3 +unk normalization, build vocab using only training data
* v2.4 +noisy training (+corr in MC)

## MD Outout format
```
[
  {
    "input": ["<sent>", ...,"</sent>"],
    "predict": ["corr", ...],
    "truth": [],
    "prob": [[0.5], [0.5], ...]
  }
]
```

## TODOs
1. Get Data
2. Normalise/clean the data
3. Split into train/test/dev and out-of-domain evaluation
4. Set up Evaluation Metric
    * WER
    * GLEU
5. Determine Baseline
    * Hunspell
    * PyThaiNLP

    * read-to-use
    * re-train on the data
6. Train Misspelling Detection
    * Bi-LSTM + Word Embedding
        * Oracle
        * Sentencepiece (No pre-tokenization required)
        * Deepcut
    * Bi-LSTM + Character Embedding
        * No pre-tokenization required
    * Bi-LSTM + Word & Charecter Combined Embedding

    * random initialisd embedding
    * initialisd embedding via pre-trained
    * frozen embedding via pre-trained
7. Build Training Pipeline
8. Train Misspelling Correction
    * Simple Seq2Seq Model
        * Machine Translation approach (E2E)
        * Only Misspelling Detection
    * BiDAF (+context attention)
9. Data Augmentation?
10. Evaluation on downstream tasks
    * Sentiment Analysis (informal)
    * Topic Classification (formal)
        * expect only tiny improvement
    * Toxicity Classification (informal)
    * Named Entity Recognition (no context needed?)

11. Integrate with MAE
12. Integrate with MST


# Author
Pakawat Nakwijit
PhD Student @ QMUL