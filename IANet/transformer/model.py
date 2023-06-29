import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ACLFTransformer(nn.Module):
    def __init__(self, d_model, vocab, d_ff=2048, dropout=0.1, concat=False):
        super(ACLFTransformer, self).__init__()
        self.d_model = d_model
        self.embed_pos1_1 = nn.Sequential(Embeddings(512, vocab), PositionalEncoding(512, dropout))
        self.embed_pos1_2 = nn.Sequential(Embeddings(512, vocab), PositionalEncoding(512, dropout))
        self.encoder1 = Encoder(num_layers=3, d_model=512, nhead=8, d_ff=d_ff, dropout=dropout)
        self.encoder2 = Encoder(num_layers=3, d_model=512, nhead=8, d_ff=d_ff, dropout=dropout)
        # self.linear = nn.Linear(512, 256)
        # self.linear1 = nn.Linear(256, 512)
        # self.linear2 = nn.Linear(256, 512)
        self.embed_pos2 = nn.Sequential(Embeddings(512, vocab), PositionalEncoding(512, dropout))
        self.decoder = Decoder(num_layers=4, d_model=512, nhead=8, d_ff=d_ff, dropout=dropout, concat=concat)
        self.generator = Generator(512, vocab)
        self.parameters_init()

    def forward(self, src, src2, tgt, src_mask, src2_mask, tgt_mask):
        x1, x2 = self.encode(src=src, src_mask=src_mask, src2=src2, src_mask2=src2_mask)
        # x1 = torch.cat([x1, x1], dim=2)
        # x2 = torch.cat([x2, x2], dim=2)
        # x1 = self.linear1(x1)
        # x2 = self.linear2(x2)
        x = self.decode(memory=x1, src_mask=src_mask, memory2=x2, src_mask2=src2_mask, tgt=tgt, tgt_mask=tgt_mask)
        return self.generator(x)

    def encode(self, src, src_mask, src2, src_mask2):
        #  x, mask
        return self.encoder1(x=self.embed_pos1_1(src), mask=src_mask), self.encoder2(x=self.embed_pos1_2(src2), mask=src_mask2)

    def decode(self, memory, src_mask, memory2, src_mask2, tgt, tgt_mask):
        # x, memory, src_mask, memory2, src_mask2, tgt_mask
        return self.decoder(x=self.embed_pos2(tgt), memory=memory, src_mask=src_mask,
                            memory2=memory2, src_mask2=src_mask2, tgt_mask=tgt_mask)
    def parameters_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, double=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.double = double

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


### encoder
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, num_layers=6, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout), num_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, nhead, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(nhead, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, num_layers=6, d_model=512, nhead=8, d_ff=2048, dropout=0.1, concat=False):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout, concat=concat), num_layers)
        self.norm = LayerNorm(d_model)

    def forward(self, x, memory, src_mask, memory2, src_mask2, tgt_mask):
        for layer in self.layers:
            x = layer(x=x, memory=memory, src_mask=src_mask, memory2=memory2, src_mask2=src_mask2, tgt_mask=tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, nhead, d_ff, dropout, concat=False):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.concat = concat

        self.self_attn = MultiHeadedAttention(nhead, d_model)
        self.src_attn1 = MultiHeadedAttention(nhead, d_model)
        self.src_attn2 = MultiHeadedAttention(nhead, d_model)
        self.lay_norm = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 4)

    def forward(self, x, memory, src_mask, memory2, src_mask2, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x1 = self.sublayer[1](x, lambda x: self.src_attn1(x, memory, memory, src_mask))
        x2 = self.sublayer[2](x, lambda x: self.src_attn2(x, memory2, memory2, src_mask2))
        x = self.lay_norm(x1 + x2)
        return self.sublayer[3](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
