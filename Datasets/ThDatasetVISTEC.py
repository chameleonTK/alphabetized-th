if __name__ == '__main__':
    import sys
    sys.path.append('../')

from Datasets.ThDataset import ThDataset
# import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

class ThDatasetVISTEC(ThDataset):
    def __init__(self, dir, name="VISTEC", cache=True):
        super().__init__(dir, name)
        self.load_data(cache=cache)
    
    def _process_dom(self, dom):
        tokens = []
        if dom.name is None:
            words = str(dom).split("|")
            
            for w in words:
                w = w.strip()
                if len(w)==0:
                    continue
                tokens.append((w, w))

        elif dom.name=="ne":
            words = dom.text.strip().split("|")
            for w in words:
                w = w.strip()
                if len(w)==0:
                    continue
                tokens.append((w, w))

        elif dom.name=="msp":
            m = dom.text.replace("|", "").strip()
            c = dom["value"].strip()
            # assert("|" not in m)

            if self._ignore(m, c):
                tokens.append((m, m))
            else:
                tokens.append((m, c))
        elif dom.name=="compound":
            for child in dom.children:
                tkn = self._process_dom(child)
                tokens += tkn
        elif dom.name=="sp":
            tokens.append((" ", " "))
        else:
            print(dom)
            raise(f"Unknown Tag: {dom.name}")

        return tokens
    
    # Ignore some orthological misspellings;
    def _ignore(self, m, c):
        if c.endswith(" ๆ"):
            return True

        if m.replace(" ", "")==c.replace(" ", ""):
            return True

        if ("ฯ" in c) and (m not in ["ๆลๆ", "พณฯท่าน", "ฯล"]):
            return True

        if ("." in m) or ("." in c):
            return True
        
        return False

    def read(self, split):
        data = []
        cnt = 0
        with open(f"{self.basedir}/{split}/VISTEC-TP-TH-2021_{split}_proprocessed.txt", encoding="utf-8") as text_file:
            with tqdm(total=40000) as pbar:

                for line in text_file:
                    line = line.strip()
                    line = line.replace("| |", "|<sp></sp>|")
                    line = line.replace("| ", "|<sp></sp>")
                    line = line.replace(" |", "<sp></sp>|")
                    s = BeautifulSoup("<div id='text'>"+line+"</div>", 'html.parser')
                    tokens = []
                    for dom in s.find("div", {"id": "text"}).children:
                        _tkn = self._process_dom(dom)
                        tokens += (_tkn)

                    words = []
                    labels = []
                    corr = []
                    for w, c in tokens:
                        
                        words.append(w)
                        corr.append(c)
                        if w==c:
                            labels.append("corr")
                        else:
                            labels.append("misp")

                    data.append({
                        "sent": "".join(words),
                        "misp": words,
                        "labels": labels,
                        "corr": corr
                    })

                    cnt += 1
                    # print(cnt)
                    pbar.update(1)
                    # if cnt % 1000:
                    #     print(cnt)
                    # break
        return data

if __name__ == '__main__':

    dataset = ThDatasetVISTEC("../Data/VISTEC-TP-TH-sample", name="VISTEC-sample")
    for d in dataset.datasets["test"]:
        print(d.keys())
        break
    