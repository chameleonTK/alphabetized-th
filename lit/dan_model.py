from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
import numpy as np
from _dan import DAN

import torch
from gensim.models import FastText

import pythainlp

class DANModel(lit_model.Model):
  """Wrapper for a Natural Language Inference model."""

  LABELS = ['negative', 'neutral', 'positive']

  def __init__(self, name, model_path, tokenizer, wv, **kw):
    # Load the model into memory so we're ready for interactive use.
    self.model = DAN.load_model(model_path)
    self.model.eval()

    self.tokenizer = tokenizer
    self.wv = wv
    
  
  # LIT API implementations
  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""

    tokens = [self.tokenizer(s["sentence"]) for s in inputs]
    token_length = [len(t) for t in tokens]
    max_token_length = max(token_length)

    W2V_SIZE = self.wv.vector_size
    vectors = []
    for sent in tokens:
        vec = []
        for w in sent:
            vec.append(self.wv.wv[w])
        
        while len(vec) < max_token_length:
            vec.append(torch.zeros(W2V_SIZE))

        vectors.append(vec)

    tensor = torch.FloatTensor(vectors)
    batch_size, nwords, _ = tensor.shape
    tensor = torch.reshape(tensor, (nwords, batch_size, W2V_SIZE))
    
    with torch.no_grad():
        answer = self.model.forward_no_embed(tensor)

    prob = torch.nn.functional.softmax(answer, dim=-1)
    

    # labels = [i["label_index"] for i in inputs]
    # labels = torch.IntTensor(labels)
    
    predited_labels = torch.max(answer, 1)[1].view(len(inputs))

    # print(predited_labels, labels)
    # n_correct = (predited_labels==labels).sum().item()
    

    
    for idx, o in enumerate(answer):

        output = {
            "probas": prob[idx].numpy(),
            "tokens": tokens[idx],
            "emb": tensor[:, idx, :].mean(dim=0)
        }

    #   output["tokens"] = []
        yield output

  def input_spec(self):
    """Describe the inputs to the model."""
    return {
        'sentence': lit_types.TextSegment(),
    }

  def output_spec(self):
    """Describe the model outputs."""
    return {
      # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
      'probas': lit_types.MulticlassPreds(vocab=self.LABELS, parent='label'),
      "tokens": lit_types.Tokens(),
      "emb": lit_types.Embeddings()
    }

if __name__ == '__main__':

    def tokenizer(text):
        text = text.lower()
        text = pythainlp.util.normalize(text)
        text = pythainlp.tokenize.word_tokenize(text)
        return text

    wv = FastText.load(f"../wv/word_th_w2v.model")
    m = DANModel("word_th", "../models/basic/word.pt", tokenizer, wv)

    
    m.predict_minibatch([
        {"sentence":"แมวกินปลา", "label": "positive", "label_index": 0},
        {"sentence":"ปลากินแมว", "label": "negative", "label_index": 1},
        {"sentence":"แมวสีเทาเดิมต๊อกแต๊ก", "label": "neutral", "label_index": 2}
    ])

    print("DONE")