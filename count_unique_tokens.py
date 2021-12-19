import sys
import tokenizer as tkn
from util import Util


if __name__ == "__main__":

    base_dir = "/import/cogsci/tk/oscar/raw_th/"
    filenames = [
        "th_part_1.txt",
        "th_part_2.txt",
        "th_part_3.txt",
        "th_part_4.txt",
        "th_part_5.txt",
    ]

    tokenizer = tkn.Tokenizer()
    
    nsent = 0
    done = False

    thwords = set()
    thtccwords = set()
    for fname in filenames:
        print("Loading text ", fname)
        with open(base_dir+fname) as fin:
            for line in fin:
                
                thline = tokenizer.subwordTokenize(line)
                thwords.update(thline)
            
                thtccline = tokenizer.subwordTCCTokenize(line)
                thtccwords.update(thtccline)
                nsent += 1
                if nsent % 10000 == 0:
                    print(nsent, len(thwords), len(thtccwords))
                    sys.stdout.flush()
                if nsent % 100000 ==0:
                    
