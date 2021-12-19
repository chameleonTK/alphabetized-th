
import sentencepiece as spm
import re
from util import Singleton
import pythainlp
from util import Util
import nltk

class Tokenizer(metaclass=Singleton):
    _models = {}
    
    def __init__(self):
        self.util = Util()

    def get_sentencepiece_model(self, key):
        if key in self._models:
            return self._models[key]

        if key=="th":
            sp = spm.SentencePieceProcessor()
            sp.Load("spm/spm_word_5M.model")
            self._models[key] = sp
        elif key=="thtcc":
            sp = spm.SentencePieceProcessor()
            sp.Load("spm/spm_word_tcc_5M.model")
            self._models[key] = sp
        # elif key=="en":
        #     sp = spm.SentencePieceProcessor()
        #     sp.Load("subword_tokenizers/subw_en10000.model")
        #     self._models[key] = sp
        else:
            raise Exception(f"Cann't load sentencepirce model: {key}")

        return self._models[key]

    def normalise(self, text):
        text = pythainlp.util.normalize(text)
        return text

    def subwordTokenize(self, text, normalised=True):
        if normalised:
            text = self.normalise(text)
        
        text = self.util.to_zh(text)
        sp = self.get_sentencepiece_model("th")
        text = sp.encode(text, out_type=str)
        text = self.util.to_th(" ".join(text)).split(" ")
        
        return text

    def subwordTCCTokenize(self, text, normalised=True):
        if normalised:
            text = self.normalise(text)
        text = self.util.tcc_encode(text)
        text = self.util.to_zh(text)

        sp = self.get_sentencepiece_model("thtcc")
        text = sp.encode(text, out_type=str)
        
        text = self.util.to_th(" ".join(text)).split(" ")
        return text

    def characterTokenize(self, text, normalised=True):
        if normalised:
            text = self.normalise(text)
        text = text.replace(" ", "_")
        return list(text)
    
    def characterTCCTokenize(self, text, normalised=True):
        if normalised:
            text = self.normalise(text)
        text = self.util.tcc_encode(text)
        text = text.replace(" ", "_")
        return list(text)

    def wordTokenize(self, text, normalised=True):
        if normalised:
            text = self.normalise(text)
        text = pythainlp.tokenize.word_tokenize(text)
        return text

    def wordTCCTokenize(self, text, normalised=True):
        if normalised:
            text = self.normalise(text)
        text = pythainlp.tokenize.word_tokenize(text)
        text = self.util.tcc_encode(" ".join(text)).split(" ")
        return text

    def wordEnTokenize(self, text, normalised=True):
        text = nltk.tokenize.word_tokenize(text)
        return text

    def subwordEnTokenize(self, text, normalised=True):
        sp = self.get_sentencepiece_model("en")
        text = sp.encode(text, out_type=str)
        return text


if __name__ == "__main__":
    # execute only if run as a script
    tokenizer = Tokenizer()
    s = "แมวกินปลา"

    print(tokenizer.subwordTCCTokenize(s))
    print(tokenizer.subwordTokenize(s))
