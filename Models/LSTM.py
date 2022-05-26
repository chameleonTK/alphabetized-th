import torch
import torch.nn as nn


class MispDetectionLSTM(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 embedding_dim = 64, 
                 hidden_dim = 64,  
                 n_layers = 2, 
                 bidirectional = True, 
                 dropout = 0.1,
                 embed_weights = None,
                 token_pad_idx = None,
                 device = None):
        
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = token_pad_idx)
        
        if embed_weights is not None:
            # embed_weights = inputs.vocab.vectors
            self.embedding.weight.data.copy_(embed_weights)
            self.embedding.weight.requires_grad = False

        self.embedding.weight.data[token_pad_idx] = torch.zeros(embedding_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tokens):
        # pass text through embedding layer
        embedded = self.dropout(self.embedding(tokens))
        # print(batch.misp.shape) [sent len, batch size]
        # print(embedded.shape) [sent len, batch size, emb dim]
        
        #pass embeddings into LSTM
        batch_size = tokens.shape[1]
        hidden = self.initHidden(batch_size)
        outputs, _ = self.lstm(embedded, hidden)
        # outputs = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)
        # print(outputs.shape) [sent len, batch size, hid dim * n directions]

        # outputs !+ hidden if nlayer > 1

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        # print(predictions.shape) [sent len, batch size, output dim]
        return predictions
    
    def initHidden(self, batch_size):
        if self.bidirectional:
            return (torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device), torch.zeros(2*self.n_layers, batch_size, self.hidden_dim).to(self.device))
        else:
            return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device))