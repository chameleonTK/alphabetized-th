from Datasets.BaseTokenizer import BaseTokenizer
from transformers import CamembertTokenizer

class SubwordTokenizer(BaseTokenizer):
    def __init__(self, dataset, **kwargs):
        self.name = "subword"

        tokenizer_dir = "airesearch/wangchanberta-base-att-spm-uncased"
        tokenizer = CamembertTokenizer.from_pretrained(tokenizer_dir)
        self.tokenizer = tokenizer
        self.space_token = "▁"
        self.subword_unk_token = "<unk>"

        self.misp_token = dataset.misp_token
        self.corr_token = dataset.corr_token

    def get_config(self):
        return {
            "kwargs": {},
            "module": "Datasets.SubwordTokenizer",
            "class": "SubwordTokenizer",
        }

    def tokenize(self, sents):
        sents = [s.lower() for s in sents]
        sentTokens = self.tokenizer(
            sents,
            max_length=None,
            truncation=False,
            # padding='max_length'            
        )

        tokens = []
        for sidx, s in enumerate(sents):
        
            _tokens = sentTokens["input_ids"][sidx]
            sent = s.replace(" ", "")

            subwords = self.tokenizer.convert_ids_to_tokens(_tokens)
            subwords = self._normalise_subtokens(sent, subwords)
            tokens.append(subwords)

        return tokens

    def tokenizeWithLabel(self, sents, sentLabels):
        tokens = []
        tokenLabels = []

        fullSents = ["".join(s).lower() for s in sents]
        sentTokens = self.tokenizer(
            fullSents,
            max_length=None,
            truncation=False,
            # padding='max_length'            
        )

        for sidx, (words, labels) in enumerate(zip(sents, sentLabels)):
            assert(len(words)==len(labels))

            _tmpTokens = []
            _tmpTokenLabels = []
            
            _tokens = sentTokens["input_ids"][sidx]
            sent = "".join(words).replace(" ", "")

            subwords = self.tokenizer.convert_ids_to_tokens(_tokens)
            subwords = self._normalise_subtokens(sent, subwords)
            assert(self.subword_unk_token not in subwords)

            _charlabels = []
            for w, l in zip(words, labels):
                for _ in w.replace(" ", ""):
                    _charlabels.append(l)

            s = 0
            for token in subwords:
                possibleLabels = _charlabels[s:s+len(token)]
                if self.misp_token in possibleLabels:
                    _tmpTokenLabels.append(self.misp_token)
                else:
                    _tmpTokenLabels.append(self.corr_token)

                _tmpTokens.append(token)
                
                s += len(token)
            
            tokens.append(_tmpTokens)
            tokenLabels.append(_tmpTokenLabels)

        return tokens, tokenLabels
    
    def _normalise_subtokens(self, sent, subwords):
        subwords = subwords[1:-1]
        norm = []
        sent = sent.replace(" ", "")
        for token in subwords:
            # remove _ from sentencepiece
            token = token.replace(self.space_token, "")
            if len(token)==0:
                continue
            
            # skip consecutive <unk>
            if token==self.subword_unk_token and len(norm)>0 and norm[-1]==self.subword_unk_token:
                continue
                
            norm.append(token)
            
        currsent = sent.lower()
        nounk = []

        error = None
        for i, token in enumerate(norm):
            if token==self.subword_unk_token:
                prevsent = "".join(nounk)
                
                if i+1 < len(norm):
                    unk = ""
                    initsent = currsent
                    while len(currsent) > 0 and (not currsent.startswith(norm[i+1])):
                        c = currsent[0]
                        unk += c
                        currsent = currsent[1:]
                        if len(currsent)==0:
                            print(sent)
                            print(norm)
                            print(initsent)
                            assert(False)

                    nounk.append(unk)
                        
                else:
                    nounk.append(currsent)        
            else:
                currsent = currsent[len(token):]
                nounk.append(token)  
                  
        assert(len(nounk)==len(norm))
        
        cased = []
        s = 0
        for token in nounk:
            casedToken = sent[s:s+len(token)]
            cased.append(casedToken)
            s = s + len(token)
        
        assert("".join(cased)==sent)
        return cased

    def _subwordidxtotokens(self, sent, tokens, sent_tokens = False):
        subwords = self.tokenizer.convert_ids_to_tokens(tokens)

        sent = "".join(sent).replace(" ", "")
        if sent_tokens:
            subwords = [self.ssent_token]+self._normalise_subtokens(sent, subwords)+[self.ssent_token]
        else:
            subwords = self._normalise_subtokens(sent, subwords)
        
        return subwords