import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
from torch.utils.data import TensorDataset, DataLoader

from lcpfn import lcpfn

# Assuming `data` is your [100, 100] tensor
input_length = 50  # Number of input elements
target_length = 50  # Number of target elements (to predict)

get_batch_func = lcpfn.create_get_batch_func(prior=lcpfn.sample_from_prior)
X, Y, Y_noisy = get_batch_func(batch_size=100, seq_len=100, num_features=1)

# Splitting data into input and target
inputs = Y[:, :input_length]
targets = Y[:, input_length : input_length + target_length]

# Create TensorDataset and DataLoader
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(
    dataset, batch_size=10, shuffle=True
)  # Adjust batch_size as needed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        self.encoder = nn.Linear(input_dim, model_dim)
        self.decoder = nn.Linear(model_dim, 50)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


model = TransformerModel(
    input_dim=1,
    model_dim=512,
    num_heads=8,
    num_layers=6,
    dim_feedforward=2048,
    output_dim=target_length,
)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for input_sequence, target_sequence in data_loader:
        optimizer.zero_grad()
        # Reshaping input to match the model's expected input shape
        input_sequence = input_sequence.unsqueeze(
            -1
        )  # Reshape to [batch_size, sequence_length, input_dim]
        output = model(input_sequence)
        loss = criterion(output, target_sequence)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")
