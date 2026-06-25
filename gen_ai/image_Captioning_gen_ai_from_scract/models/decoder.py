import torch
import torch.nn as nn
from .lstm_cell import LSTMCell


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocal_size, feature_size = 2048):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocal_size, embed_size)

        self.projection = nn.Linear(feature_size, hidden_size)

        self.lstm_cell = LSTMCell(embed_size, hidden_size)

        self.fc = nn.Linear(hidden_size, vocal_size)

    def forward(self, features, captions):
        batch_size = features.size(0)

        h_prev = self.projection(features)

        C_prev = torch.zeros_like(h_prev)

        embeddings = self.embed(captions)

        outputs = []

        for t in range(captions.shape[1] - 1):
            current_embed = embeddings[:, t, :]

            h_prev, C_prev = self.lstm_cell(current_embed, h_prev, C_prev)

            out = self.fc(h_prev)

            outputs.append(out)
        
        return torch.stack(outputs, dim = 1)