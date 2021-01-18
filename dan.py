import torch
import torch.nn as nn
import torch.nn.functional as F


class DAN(nn.Module):

    def __init__(self,
                 n_embed=10000,
                 d_embed=300,
                 d_hidden=256,
                 d_out=2,
                 dp=0.2,
                 embed_weight=None):
        super(DAN, self).__init__()

        self.n_embed = n_embed
        self.d_embed = d_embed
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.dp = dp

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embed = nn.Embedding(n_embed, d_embed)

        if embed_weight is not None:
            # embed_weight = inputs.vocab.vectors
            self.embed.weight.data.copy_(embed_weight)
            self.embed.weight.requires_grad = False

        self.dropout1 = nn.Dropout(dp)
        self.bn1 = nn.BatchNorm1d(d_embed)
        self.fc1 = nn.Linear(d_embed, d_hidden)

        self.dropout2 = nn.Dropout(dp)
        self.bn2 = nn.BatchNorm1d(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, batch):
        text = batch.norm_text
        label = batch.label

        x = self.embed(text)

        x = x.mean(dim=0)

        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x

    def save_model(self, model_path):
        torch.save({
            "model_state_dict": self.state_dict(),
            "n_embed": self.n_embed,
            "d_embed": self.d_embed,
            "d_hidden":self.d_hidden,
            "d_out": self.d_out,
            "dp": self.dp,
        }, model_path)

    @staticmethod
    def load_model(model_path):
        checkpoint = torch.load(model_path)
        newmodel = DAN(n_embed=checkpoint["n_embed"], 
                    d_embed=checkpoint["d_embed"], 
                    d_hidden=checkpoint["d_hidden"],
                    d_out=checkpoint["d_out"],
                    dp=checkpoint["dp"])
        newmodel.load_state_dict(checkpoint["model_state_dict"])
        newmodel.eval()

        return newmodel