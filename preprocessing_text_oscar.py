import tokenizer as tkn
from util import Util

### set data for building word vectors
sents = []
name = "../oscar/oscar"

ntoken1, ntoken2 = 0, 0
nsent = 0
filenames = [
  "../oscar/text/th_part_1.txt",
  "../oscar/text/th_part_2.txt",
  "../oscar/text/th_part_3.txt",
  "../oscar/text/th_part_4.txt",
  "../oscar/text/th_part_5.txt",
  "../oscar/text/th_part_6.txt",
  "../oscar/text/th_part_7.txt",
  "../oscar/text/th_part_8.txt",
  "../oscar/text/th_part_9.txt",
]

tokenizer = tkn.Tokenizer()
idx = 0
for filename in filenames:  
  print(filename)
  idx += 1
  if idx in [3,4,5,6,7]:
     continue
  fout1 = open(f"{name}_word_th{idx}.txt", "w", encoding="utf-8")
  fout2 = open(f"{name}_word_th_tcc{idx}.txt", "w", encoding="utf-8")
  with open(filename) as fin:
    for text in fin:
        #if len(text) > 2000:
        #  continue
        
        if len(text) < 10:
          continue

        text = text.strip()
        text1 = tokenizer.wordTokenize(text)
        text2 = tokenizer.wordTCCTokenize(text)
        #print(text1, text2)
        #break

        fout1.write(" ".join(text1)+"\n")
        fout2.write(" ".join(text2)+"\n")
        
        ntoken1 += len(text1)
        ntoken2 += len(text2)
        nsent += 1

        if nsent%100000==0:
          print(nsent, ntoken1, ntoken2, len(str(ntoken1)))

  print(f"DONE {filename}")
  print(nsent, ntoken1, ntoken2, len(str(ntoken1)))
  fout1.close()
  fout2.close()

print("Number of Token", ntoken1, ntoken2)
print("Number of Sentence", nsent)
