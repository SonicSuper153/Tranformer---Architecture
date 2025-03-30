import torch
import torch.nn as nn
import math

#creating embeddings of the sentence which is a vector of 512 size
class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocal_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocal_size = vocal_size
        self.embedding = nn.Embedding(vocal_size, d_model)

    def forward(self,x):
        self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):

    def __init__(self,d_model: int, seq_len: int , dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        #a matrix of seq length
        pe = torch.zeros(seq_len, d_model) 

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self,eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # value is 1
        self.beta = nn.Parameter(torch.zeros(1)) # value is 0
    
    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardLayer(nn.Module):
    def __init__(self,d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.linear(d_model,d_ff):
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.linear(d_ff,d_model)
    
    def forward(self,x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadttentionBlock(nn.Module):

    def __init__(self, d_model:int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
