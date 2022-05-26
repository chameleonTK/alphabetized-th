
import json
import torch
# from Evaluation.WER import WER

import random
class EvaluationScore:
    def __init__(self, values, eval_metric="f1"):
        self.values = values
        self.eval_metric = eval_metric

    def get_values(self):
        return self.values

    def get_value(self, k):
        v = self.values[k]
        if torch.is_tensor(v):
            return v.item()
        
        return v

    def get_eval_value(self):
        return self.get_value(self.eval_metric)
        
    def __lt__(self, other):
        return self.values[self.eval_metric] < other.values[self.eval_metric]

    def __repr__(self):
         return f"Evaluation({self.eval_metric}={self.values[self.eval_metric]})"

    def __str__(self):
         return f"Evaluation({self.eval_metric}={self.values[self.eval_metric]})"

class Evaluation:
    def __init__(self, datasets, model_selection="f1"):
        labels = datasets.fields["labels"]
        LABEL_PAD_IDX = labels.vocab.stoi[labels.pad_token]
        self.labels = labels
        self.pad_idx = LABEL_PAD_IDX
        self.model_selection = model_selection
        
    def eval(self, results):
        pass
    
    def calLoss(self, batch, model, criterion):
        pass

    def run(self, model, iterator, criterion, return_acc=True, return_pred=False, eval_sample=None):
        epoch_loss = 0
        model.eval()
        results = []
        # print("##### start eval #####")
        # if eval_sample is not None:
        #     print("Eval sample:", eval_sample)

        with torch.no_grad():
            eval_count = 0
            for batch in iterator:
                if eval_sample is not None:
                    if eval_count >= eval_sample*len(batch):
                        continue

                    elif eval_sample < random.random():
                            continue
                    else:
                        eval_count += 1

                loss, r = self.calLoss(batch, model, criterion)
                epoch_loss += loss
                results.append(r)

        acc = None
        if return_acc:
            acc = self.eval(results)

        # print("##### end eval #####")
        if return_pred:
            return epoch_loss / len(iterator), acc, results 

        return epoch_loss / len(iterator), acc