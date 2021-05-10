import torch
import torch.nn as nn
import math

def fourier_encode(x, max_freq, num_bands = 4, base = 2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(0., math.log(max_freq / 2) / math.log(base), num_bands, base = base, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, dim_head, num_heads=1, context_dim = None, dropout=0):
        super().__init__()
        layers = []
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class VisionPerceiver(nn.Module):
    def __init__(self, input_dim, max_freq = 8,
                 input_channels=3, depth=1,
                 num_latents=32, latent_dim=128,
                 cross_heads = 1, cross_dim_head = 8,
                 latent_heads = 2, latent_dim_head = 8,
                 num_classes = 10,
                 attn_dropout = 0., ff_dropout = 0.):

        super().__init__()

        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.max_freq = max_freq
        input_channels *= 9
        # perceiver stacks
        self.layers = nn.ModuleList([])
        for i in range(depth): # build each perceiver cell
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
            cell.append(nn.LayerNorm(latent_dim))
            cell.append(MultiHeadAttention(latent_dim, dim_head = latent_dim_head,
                                           num_heads = latent_heads, dropout = attn_dropout))
            # feed forward
            cell.append(nn.LayerNorm(latent_dim))
            cell.append(FeedForward(latent_dim, dropout = ff_dropout))

            self.layers.append(cell)

        self.to_logits = nn.Sequential(
                            nn.LayerNorm(latent_dim),
                            nn.Linear(latent_dim, num_classes))

    def forward(self, data, attn_mask=None, latent_init = None, seed = None):
        # flatten
        N, C, H, W = data.shape
        data = data.view(N, C, -1).transpose(1, 2)

        # encoding
        data = fourier_encode(data, self.max_freq)

        # determine the initial latent vector
        if latent_init is None:
            if seed is not None:
                torch.manual_seed(seed)
            latent_init = torch.randn(self.num_latents, self.latent_dim).unsqueeze(0).repeat([N, 1, 1])
            latent_init.requires_grad = False
        else:
            assert latent_init.shape == (num_latents, latent_dim)
            latent_init.unsqueeze(0).repeat([N, 1, 1])

        self.latent_init = nn.Parameter(latent_init)

        x = latent_init
        for cell in self.layers:
            # cross attention
            y = cell[1](data.reshape(N, -1))
            x = cell[2](cell[0](x), y.reshape(N, H*W, -1), attn_mask=attn_mask) + x
            # feed forward
            x = cell[4](cell[3](x)) + x
            # latent transformer
            x = cell[6](cell[5](x), attn_mask=attn_mask) + x
            # feed forward
            x = cell[8](cell[7](x)) + x

        x = x.mean(dim = -2)
        return self.to_logits(x)
