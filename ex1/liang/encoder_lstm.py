import torch
import torchvision
import torch.nn as nn


class autoencoder_lstm(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_out):
        super(autoencoder_lstm, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_out)
        )

    def forward(self, x):
        encode, _ = self.encoder(x)
        encode = encode[:, -1, :]
        decode = self.decoder(encode)
        return decode, encode