import torch
from Evaluation.Evaluation import Evaluation, EvaluationScore
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

class EvaluationMD(Evaluation):
    def __init__(self, datasets, window=None, model_selection="f1", binary_class=False):
        super().__init__(datasets, model_selection)
        self.window = window
        self.misp_token = datasets.misp_token
        self.misp_idx = self.labels.vocab.stoi[datasets.misp_token]

        self.corr_token = datasets.corr_token
        self.corr_idx = self.labels.vocab.stoi[datasets.corr_token]

        self.binary_class = binary_class

    def eval(self, results):
        prediction = torch.cat([a[1].view(-1) for a in results])
        ground_truth = torch.cat([a[2].view(-1) for a in results])

        pad_idx = self.pad_idx
        idx = (ground_truth != pad_idx).nonzero()
        
        p = prediction[idx].cpu()
        p[p==pad_idx] = self.corr_idx

        y = ground_truth[idx].cpu()
        

        label_names = [self.misp_token, self.corr_token]
        labels = self.labels.vocab.stoi
        lidx = [labels[k] for k in label_names]

        mat = confusion_matrix(y, p, labels=lidx).tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(y, p, labels=lidx, average=None)
    
        return EvaluationScore({
            "f1": f1[0],
            "precision": precision[0],
            "recall": recall[0],
            "tp": mat[0][0],
            "fp": mat[0][1],
            "fn": mat[1][0],
            "tn": mat[1][1],
            
            
            "labels": label_names,
            "confusion_matrix": mat
        }, eval_metric=self.model_selection)

    def calLoss(self, batch, model, criterion, thredhold=0.5):
        window = self.window
        nword, nsent = batch.misp.shape
        
        # If window is None, treat as one big window
        if window is None:
            window = nword

        # If window is bigger than nword
        if nword < window:
            window = nword

        loss = 0
        nwindow = nword-window+1
        
        outputs = torch.zeros(nword, nsent, nwindow)
        prob = torch.zeros(nword, nsent, 1)
        

        for i in range(nwindow):
            tokens = batch.misp[i:i+window, :]
            labels = batch.labels[i:i+window, :]

            predictions = model(tokens)

            if self.binary_class:
                prob[i:i+window, :, :] += F.sigmoid(predictions).detach().cpu()
                
                p = predictions.view(-1)
                g = labels.view(-1)

                if criterion is not None:
                    pad_idx = self.pad_idx
                    idx = (g != pad_idx).nonzero()
                    
                    p = p[idx]
                    g = (g[idx]==self.misp_idx).float()
                    loss += criterion(p, g).item()
            else:
                p = F.softmax(predictions, dim=2).cpu()
                ppos = p[:, :, self.misp_idx]
                pneg = p[:, :, self.corr_idx]
                prob[i:i+window, :, :] = (ppos/(ppos+pneg)).unsqueeze(dim=2)
                
                # assert(False)
                N = predictions.shape[-1]
                p = predictions.view(-1, N)
                g = labels.view(-1)
                if criterion is not None:
                    loss += criterion(p, g).item()
                
                # predictions = predictions.argmax(dim = 2, keepdim = True).view(tokens.shape[0], -1)

            predictions = torch.zeros(window, nsent, 1)
            predictions[(prob[i:i+window, :, :] > thredhold)] = self.misp_idx
            predictions[(prob[i:i+window, :, :] <= thredhold)] = self.corr_idx
            predictions = torch.squeeze(predictions, dim=2)
            
            outputs[i:i+window, :, i] = predictions

        predictions = torch.zeros(nword, nsent)
        for i in range(nword):
            o = outputs[i, :, max(0, i-window+1):(i+1)]
            mod = o.mode(dim=1)
            predictions[i, :] = mod.values
            prob[i, :, :] = prob[i, :, :] / o.shape[1]
            
        loss = loss/nwindow
        return loss, (batch.misp, predictions, batch.labels, prob)