# Lint as: python3
r"""Code example for a custom model, using PyTorch.

This demo shows how to use a custom model with LIT, in just a few lines of code.
We'll use a transformers model, with a minimal amount of code to implement the
LIT API. Compared to models/glue_models.py, this has fewer features, but the
code is more readable.

This demo is equivalent in functionality to simple_tf2_demo.py, but uses PyTorch
instead of TensorFlow 2. The models behave identically as far as LIT is
concerned, and the implementation is quite similar - to see changes, run:
  git diff --no-index simple_tf2_demo.py simple_pytorch_demo.py

The transformers library can load weights from either,
so you can use any saved model compatible with the underlying model class
(AutoModelForSequenceClassification). To train something for this demo, you can:
- Use quickstart_sst_demo.py, and set --model_path to somewhere durable
- Or: Use tools/glue_trainer.py
- Or: Use any fine-tuning code that works with transformers, such as
https://github.com/huggingface/transformers#quick-tour-of-the-fine-tuningusage-scripts

To run locally:
  python -m lit_nlp.examples.simple_pytorch_demo \
      --port=5432 --model_path=/path/to/saved/model

Then navigate to localhost:5432 to access the demo UI.

NOTE: this demo still uses TensorFlow Datasets (which depends on TensorFlow) to
load the data. However, the output of glue.SST2Data is just NumPy arrays and
plain Python data, and you can easily replace this with a different library or
directly loading from CSV.
"""
from absl import app
from absl import flags
from absl import logging

from lit_nlp import dev_server
from lit_nlp import server_flags
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
# Use the regular GLUE data loaders, because these are very simple already.
from lit_nlp.examples.datasets import glue
from lit_nlp.lib import utils

import torch
import transformers

# NOTE: additional flags defined in server_flags.py

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", None,
    "Path to trained model, in standard transformers format, e.g. as "
    "saved by model.save_pretrained() and tokenizer.save_pretrained()")


def _from_pretrained(cls, *args, **kw):
  """Load a transformers model in PyTorch, with fallback to TF2/Keras weights."""
  try:
    return cls.from_pretrained(*args, **kw)
  except OSError as e:
    logging.warning("Caught OSError loading model: %s", e)
    logging.warning(
        "Re-trying to convert from TensorFlow checkpoint (from_tf=True)")
    return cls.from_pretrained(*args, from_tf=True, **kw)


class SimpleSentimentModel(lit_model.Model):
  """Simple sentiment analysis model."""

  LABELS = ["0", "1"]  # negative, positive

  def __init__(self, model_name_or_path):
    self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=2,
        output_hidden_states=True,
        output_attentions=True,
    )
    # This is a just a regular PyTorch model.
    self.model = _from_pretrained(
        transformers.AutoModelForSequenceClassification,
        model_name_or_path,
        config=model_config)
    self.model.eval()

  ##
  # LIT API implementation
  def max_minibatch_size(self):
    # This tells lit_model.Model.predict() how to batch inputs to
    # predict_minibatch().
    # Alternately, you can just override predict() and handle batching yourself.
    return 32

  def predict_minibatch(self, inputs):
    # Preprocess to ids and masks, and make the input batch.
    encoded_input = self.tokenizer.batch_encode_plus(
        [ex["sentence"] for ex in inputs],
        return_tensors="pt",
        add_special_tokens=True,
        max_length=128,
        pad_to_max_length=True)

    # Check and send to cuda (GPU) if available
    if torch.cuda.is_available():
      self.model.cuda()
      for tensor in encoded_input:
        encoded_input[tensor] = encoded_input[tensor].cuda()
    # Run a forward pass.
    with torch.no_grad():  # remove this if you need gradients.
      logits, embs, unused_attentions = self.model(**encoded_input)

    # Post-process outputs.
    batched_outputs = {
        "probas": torch.nn.functional.softmax(logits, dim=-1),
        "input_ids": encoded_input["input_ids"],
        "ntok": torch.sum(encoded_input["attention_mask"], dim=1),
        "cls_emb": embs[-1][:, 0],  # last layer, first token
    }
    # Return as NumPy for further processing.
    detached_outputs = {k: v.numpy() for k, v in batched_outputs.items()}
    # Unbatch outputs so we get one record per input example.
    for output in utils.unbatch_preds(detached_outputs):
      ntok = output.pop("ntok")
      output["tokens"] = self.tokenizer.convert_ids_to_tokens(
          output.pop("input_ids")[1:ntok - 1])
      yield output

  def input_spec(self) -> lit_types.Spec:
    return {
        "sentence": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=self.LABELS, required=False)
    }

  def output_spec(self) -> lit_types.Spec:
    return {
        "tokens": lit_types.Tokens(),
        "probas": lit_types.MulticlassPreds(parent="label", vocab=self.LABELS),
        "cls_emb": lit_types.Embeddings()
    }


def main(_):
  # Load the model we defined above.
  models = {"sst": SimpleSentimentModel(FLAGS.model_path)}
  # Load SST-2 validation set from TFDS.
  datasets = {"sst_dev": glue.SST2Data("validation")}

  # Start the LIT server. See server_flags.py for server options.
  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  lit_demo.serve()


if __name__ == "__main__":
  app.run(main)