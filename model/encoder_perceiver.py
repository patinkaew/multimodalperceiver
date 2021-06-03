import torch
import torch.nn as nn
import math
from model.positional_encodings import PositionalEncodingPermute2D

class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, dim_head, num_heads=1, context_dim = None, dropout=0):
        super().__init__()
        embed_dim = dim_head * num_heads
        if context_dim is None:
            context_dim = query_dim

        self.H = num_heads
        self.E = embed_dim
        self.q_linear = nn.Linear(query_dim, embed_dim)
        self.k_linear = nn.Linear(context_dim, embed_dim)
        self.v_linear = nn.Linear(context_dim, embed_dim)
        self.o_linear = nn.Linear(embed_dim, query_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, context = None, attn_mask=None):
        if context is None:
            context = query
        N, S, query_dim = query.shape
        N, T, context_dim = context.shape

        q = self.q_linear(query)
        k = self.k_linear(context)
        v = self.v_linear(context)
        q_split = q.view((N, S, self.H, int(self.E/self.H))).transpose(1, 2)
        k_split = k.view((N, T, self.H, int(self.E/self.H))).transpose(1, 2)
        v_split = v.view((N, T, self.H, int(self.E/self.H))).transpose(1, 2)
        a = torch.matmul(q_split, k_split.transpose(2, 3))/math.sqrt(self.E/self.H)
        if attn_mask is not None:
            a = a.masked_fill(~(attn_mask.type(torch.bool)), -math.inf)
        e = torch.nn.functional.softmax(a, dim=3)
        y = torch.matmul(e, v_split)
        output = self.o_linear(y.transpose(1, 2).reshape(N, S, self.E))
        return self.dropout(output)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class EncoderPerceiver(nn.Module):
    def __init__(self, input_dim, input_channels=3,
                 num_iterations = 1, num_transformer_blocks = 4,
                 num_latents = 32, latent_dim = 128,
                 cross_heads = 1, cross_dim_head = 8,
                 latent_heads = 2, latent_dim_head = 8,
                 attn_dropout = 0., ff_dropout = 0.,
                 latent_init = None, seed = None):

        super().__init__()

        # initialize latent vector
        if latent_init is None:
            if seed is not None:
                torch.manual_seed(seed)
            latent_init = torch.randn(num_latents, latent_dim).unsqueeze(0)
            latent_init.requires_grad = False
        else:
            assert latent_init.shape == (num_latents, latent_dim)
            latent_init.unsqueeze(0)

        self.latent_init = nn.Parameter(latent_init)
        self.encode_data = PositionalEncodingPermute2D(input_channels)

        # perceiver stacks
        self.layers = nn.ModuleList([])
        for i in range(num_iterations): # build each perceiver cell
            cell = nn.ModuleList([])
            # cross attention module
            cell.append(nn.LayerNorm(latent_dim))
            cell.append(nn.LayerNorm(input_dim * input_channels))
            cell.append(MultiHeadAttention(latent_dim, dim_head = cross_dim_head,
                                           num_heads = cross_heads, context_dim = input_channels,
                                           dropout = attn_dropout))
            # feed forward
            cell.append(nn.LayerNorm(latent_dim))
            cell.append(FeedForward(latent_dim, dropout = ff_dropout))

            # latent transformer
            latent_transformer = nn.ModuleList([])
            for j in range(num_transformer_blocks):
                latent_transformer_block = nn.ModuleList([])
                # self attention
                latent_transformer_block.append(nn.LayerNorm(latent_dim))
                latent_transformer_block.append(MultiHeadAttention(latent_dim, dim_head = latent_dim_head,
                                           num_heads = latent_heads, dropout = attn_dropout))
                # feed forward
                latent_transformer_block.append(nn.LayerNorm(latent_dim))
                latent_transformer_block.append(FeedForward(latent_dim, dropout = ff_dropout))
                latent_transformer.append(latent_transformer_block)
            cell.append(latent_transformer)

            self.layers.append(cell)

    def forward(self, data, attn_mask = None):
        # flatten
        N, C, H, W = data.shape
        x = self.latent_init.repeat([N, 1, 1])
        data = data + self.encode_data(data)

        for cell in self.layers:
            # cross attention
            y = cell[1](data.reshape(N, -1))
            x = cell[2](cell[0](x), y.reshape(N, H*W, -1), attn_mask=attn_mask) + x
            # feed forward
            x = cell[4](cell[3](x)) + x
            # latent transformer
            for latent_transformer in cell[5]:
                # self attention
                x = latent_transformer[1](latent_transformer[0](x), attn_mask=attn_mask) + x
                # feed forward
                x = latent_transformer[3](latent_transformer[2](x)) + x

        x = x.mean(dim = -2)
        return x # return latent as image feature
