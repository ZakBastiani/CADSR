import torch
from torch import nn, Tensor
import math


class NoPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return x


class OneDimensionalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, device=torch.device("cuda")):
        super().__init__()

        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1).to(device)
        d_model_ceil = math.ceil(d_model / 2) * 2
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device)
        self.pe = torch.zeros(1, max_len, d_model_ceil).to(device)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :, :self.d_model]
        return x


class TwoDimensionalPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 32, device=torch.device("cuda")):
        super().__init__()
        self.d_model = d_model
        self.d_model_mod_ceil = math.ceil(d_model / 4)
        self.d_model_ceil = math.ceil(d_model / 4) * 4
        self.max_length = max_len
        self.device = device
        self.div_term_10000 = torch.exp(torch.arange(0, self.d_model, 4) * (-math.log(10000.0) / self.d_model)).to(self.device)
        self.div_term_1 = torch.exp(torch.arange(0, self.d_model, 4) * (-math.log(10.0) / self.d_model)).to(self.device)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sequence_length, embedding_dim]``
        """
        pe = torch.zeros((len(x), self.max_length, self.d_model_ceil), device=self.device)
        pe[:, :, 0::4] = torch.sin(positions[:, :, 0].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_10000)
        pe[:, :, 1::4] = torch.cos(positions[:, :, 0].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_10000)
        pe[:, :, 2::4] = torch.sin(positions[:, :, 1].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_1)
        pe[:, :, 3::4] = torch.cos(positions[:, :, 1].unsqueeze(2).repeat(1, 1, self.d_model_mod_ceil) * self.div_term_1)
        x = x + pe[:, :, :self.d_model]
        return x
