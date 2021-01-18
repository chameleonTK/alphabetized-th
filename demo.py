from absl import app
from absl import flags
from absl import logging


import lit_nlp
from lit_nlp import server_flags
from lit_nlp.lib import utils

from mnli_data import MultiNLIData
from nli_model import NLIModel

from lit_nlp import dev_server

def main(_):
  # MulitiNLIData implements the Dataset API
  datasets = {
      'mnli_matched': MultiNLIData("./datasets/MNLI/multinli_1.0_dev_matched.csv"),
      'mnli_mismatched': MultiNLIData("./datasets/MNLI/multinli_1.0_dev_mismatched.csv"),
  }

  # NLIModel implements the Model API
  models = {
    'model_0': NLIModel(0),
    'model_1': NLIModel(1),
  }

  lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
  return lit_demo.serve()

if __name__ == '__main__':
  app.run(main)