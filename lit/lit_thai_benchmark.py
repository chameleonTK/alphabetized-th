from absl import app
from absl import flags
from absl import logging


import lit_nlp
from lit_nlp import server_flags
from lit_nlp.lib import utils

from wisesight_data import WisesightData
from dan_model import DANModel

from lit_nlp import dev_server
import _tokenizer as tkn

from gensim.models import FastText

def main(_):
  # MulitiNLIData implements the Dataset API
  datasets = {
      'wisesight': WisesightData("../datasets/Wisesight-sentiment/wisesight_test.csv")
  }

  tokenizer = tkn.Tokenizer()
  wv_word = FastText.load(f"../wv/word_th_w2v.model")
  wv_word_tcc = FastText.load(f"../wv/word_th_tcc_w2v.model")
  # wv_subword = FastText.load(f"../wv/subword_th_w2v.model")
  # wv_subword_tcc = FastText.load(f"../wv/subword_th_tcc_w2v.model")

  models = {
    'word_th': DANModel("word_th", "../models/basic/word.pt", tokenizer.wordTokenize, wv_word),
    'word_thtcc': DANModel("word_thtcc", "../models/basic/word_tcc.pt", tokenizer.wordTCCTokenize, wv_word_tcc),
    # 'subword_th': DANModel("subword_th", "../models/basic/subword.pt", tokenizer.subwordTokenize, wv_subword),
    # 'subword_thtcc': DANModel("subword_thtcc", "../models/basic/subword_tcc.pt", tokenizer.subwordTCCTokenize, wv_subword_tcc)
  }

  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  return lit_demo.serve()

if __name__ == '__main__':
  app.run(main)