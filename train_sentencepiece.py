import pythainlp
from util import Util
import sentencepiece as spm

def prepare_text(path):
    util = Util()
    for index in [1,2,3,4,5]:
        print("File index", index)
        
        fout_word = open(f"oscar_word_th{index}.txt", "w", encoding="utf-8")
        fout_wordtcc = open(f"oscar_word_thtcc{index}.txt", "w", encoding="utf-8")

        # ntoken = 0 
        nsent = 0
        with open(f"{path}th_part_{index}.txt") as fin:
            for text in fin:
                text = pythainlp.util.normalize(text)

                fout_word.write(util.to_zh(text)+"\n")
                fout_wordtcc.write(util.to_zh(util.tcc_encode(text, any=True))+"\n")

                # assert(len(text.split(" "))==len(tcc_encode(text).split(" ")))
                
                nsent += 1

                if nsent%100000==0:
                    print(nsent)

        fout_word.close()
        fout_wordtcc.close()
        print("DONE", index, nsent)

def create_tmp(mode=""):
    print("Loading text ")

    filenames = [
        f"oscar_spm/oscar_word_th{mode}1.txt",
        f"oscar_spm/oscar_word_th{mode}2.txt",
        f"oscar_spm/oscar_word_th{mode}3.txt",
        f"oscar_spm/oscar_word_th{mode}4.txt",
        # "oscar_spm/oscar_word_th5.txt",
        #"oscar_spm/oscar_word_th6.txt",
        #"oscar_spm/oscar_word_th7.txt",
        #"oscar_spm/oscar_word_th8.txt",
        #"oscar_spm/oscar_word_th9.txt",
    ] 

    tmp_filename = f"_tmp_merged_text{mode}.txt"
    fo = open(tmp_filename, "w", encoding="utf-8")
    nsent = 0
    done = False
    for fname in filenames:
        with open(fname, encoding="utf-8") as fin:
            for line in fin:
                fo.write(line)
                nsent += 1
                if nsent % 100000 ==0:
                    print(nsent)

                if nsent > (5e6):
                    done = True
                    break
        print(fname, nsent)
        if done:
            break
    fo.close()
    print(nsent)

import sys
if __name__ == "__main__":
    print("Train sentencepiece")
    #prepare_text("../oscar/")

    #create_tmp("tcc")
    #create_tmp()

    if len(sys.argv) > 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        print("START")
        print(sys.argv)
        spm.SentencePieceTrainer.train(f'--input={input_path} --model_prefix={output_path} --vocab_size=32000 --character_coverage=1.0')
        print("DONE")    
        
    else:
        print("Please specify wordvector location")
        sys.exit(0)

    # sp = spm.SentencePieceProcessor()
    # sp.Load("./w2v/subw10000.model")

    # util = Util()
    # text = pythainlp.util.normalize("แมวกินปลา")
    # text = util.to_zh(text)
    # text = sp.encode(text, out_type=str)
    # text = to_th(" ".join(text)).split(" ")
    
