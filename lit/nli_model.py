from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp.lib import utils
import numpy as np

class NLIModel(lit_model.Model):
  """Wrapper for a Natural Language Inference model."""

  NLI_LABELS = ['negative', 'neutral', 'positive']

  def __init__(self, model_path, **kw):
    # Load the model into memory so we're ready for interactive use.
    # self._model = _load_my_model(model_path, **kw)
    self._model = None
  
  # LIT API implementations
  def predict_minibatch(self, inputs):
    """Predict on a single minibatch of examples."""
    mock_outputs = [example for example in inputs]

    for output in utils.unbatch_preds(mock_outputs):
      output = {
        "probas": [0, 0, 1],
        # "input_ids": np.random.randn(100, 10),
        "tokens": output["premise"].split(),
        "cls_emb": np.random.randn(100)
      }

      output["tokens"] = []
      yield output
    # return self._model.predict_examples(inputs)  # returns a dict for each input

  def input_spec(self):
    """Describe the inputs to the model."""
    return {
        'premise': lit_types.TextSegment(),
        'hypothesis': lit_types.TextSegment(),
    }

  def output_spec(self):
    """Describe the model outputs."""
    return {
      # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
      'probas': lit_types.MulticlassPreds(vocab=self.NLI_LABELS, parent='label'),
      "tokens": lit_types.Tokens(),
      "cls_emb": lit_types.Embeddings()
    }