import pandas
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import torchtext

class WisesightData(lit_dataset.Dataset):
  LABELS = ['negative', 'neutral', 'positive']
  LABELS_INDEX = [0, 1, 2]

  def __init__(self, path):
    # Read the eval set from a .tsv file as distributed with the GLUE benchmark.
    df = pandas.read_csv(path)
    df["label_text"] = df.label.apply(self.transform_label)
    
    self._examples = [{
      'sentence': row['norm_text'],
      'label': row['label_text'],
      'label_num': row['label'],
    } for _, row in df.iterrows()]


  def transform_label(self, l):
    if l==0:
      return "negative"
    elif l==1:
      return "neutral"
    else:
      return "positive"

  def transform_label_text(self, l):
    if l=="negative":
      return 0
    elif l=="neutral":
      return 1
    else:
      return 2

  def spec(self):
    return {
      'sentence': lit_types.TextSegment(),
      'label': lit_types.CategoryLabel(vocab=self.LABELS),
      "label_num": lit_types.CategoryLabel(vocab=self.LABELS_INDEX),
    }

if __name__ == "__main__":

  d = WisesightData("../datasets/Wisesight-sentiment/wisesight_train.csv")