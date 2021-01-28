import tokenizer as tkn
from util import Util

### set data for building word vectors
sents = []
name = "wiki"
fout1 = open(f"{name}_word_th.txt", "w", encoding="utf-8")
fout2 = open(f"{name}_word_th_tcc.txt", "w", encoding="utf-8")

ntoken1, ntoken2 = 0, 0
nsent = 0
filenames = [
    "./datasets/data_th.txt",
]

tokenizer = tkn.Tokenizer()
nsent
for filename in filenames:  
  with open(filename) as fin:
    for text in fin:
        if len(text) > 2000:
          continue
        
        if len(text) < 100:
          continue

        text = text.strip()
        text1 = tokenizer.wordTokenize(text)
        text2 = tokenizer.wordTCCTokenize(text)
        # print(text1, text2)
        # break

        fout1.write(" ".join(text1)+"\n")
        fout2.write(" ".join(text2)+"\n")
        
        ntoken1 += len(text1)
        ntoken2 += len(text2)
        nsent += 1

        if nsent%1000000==0:
          print(nsent, ntoken1, ntoken2, len(str(ntoken1)))

  print(f"DONE {filename}")
  print(nsent, ntoken1, ntoken2, len(str(ntoken1)))

print("Number of Token", ntoken)
print("Number of Sentence", nsent)
fout1.close()
fout2.close()
