import sys
sys.path.append('../')

from Datasets.CharTokenizer import CharTokenizer
from Datasets.SubwordTokenizer import SubwordTokenizer
from Datasets.ThMCDataset import ThMCDataset
from Datasets.ThMDDataset import ThMDDataset
from Datasets.ThDatasetVISTEC import ThDatasetVISTEC
import torch

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    maindataset = ThDatasetVISTEC("../Data/VISTEC-TP-TH-sample", name="VISTEC-sample")
    for d in maindataset.datasets["test"]:
        print("INPUT", d["sent"])
        break
        if "ท่ดจ้า" in "".join(d["sent"]):
            print(d)
            

    # tokenizer = CharTokenizer(maindataset)
    tokenizer = SubwordTokenizer(maindataset)
    dataset = ThMCDataset(maindataset, tokenizer, cache=False)
    dataset.build(device)

    for d in dataset.datasets["test"]:
        for k in d:
            print(">", k, d[k])
        break

    print()
    print("Processing...")
    mddataset = ThMDDataset(maindataset, tokenizer)

    cc = 0
    for i, d in enumerate(mddataset.datasets["test"]):
        output = [
            {
                "input": d["misp"],
                "predict": d["labels"],
                "truth": [],
                "prob": [[0] if l==mddataset.corr_token else [1] for l in d["labels"]]
            }
        ]

        nd, spanidx = dataset.process(output, "cpu")

        if cc==0:
            for b in nd:
                s = ""
                for w in b.misp:
                    s += dataset.fields["tokens"].vocab.itos[w]+" "
                
                print("> OUT", s)
                break

        # Verify
        # For subword model, it might not hold true
        for b in nd:
            s = ""
            for w in b.misp:
                s += dataset.fields["tokens"].vocab.itos[w]+""
            
            o = "".join(dataset.datasets["test"][cc]["misp"])
            # print("> OUT", s)
            if (dataset.unk_token not in s) and o!=s.strip() and cc not in [14]:
                print(o)
                print(dataset.datasets["test"][cc])
                print("==============")

                print(s)
                print(mddataset.datasets["test"][i])
                # print(maindataset.datasets["test"][i])

                

                # k = i-1
                # while "misp" not in mddataset.datasets["test"][k]["labels"]:
                #     k -= 1
                # print(mddataset.datasets["test"][k])

                # print(k)
                assert(False)
                
            cc += 1
        # break
    
    