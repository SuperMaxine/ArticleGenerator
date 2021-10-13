import torch
from torch import nn
import config as cfg
class Attention(nn.Module):
    def __init__(self, isMask=True):
        super().__init__()
        self.dk = (cfg.embed_dim // cfg.head_num) ** 0.5
        self.isMask = isMask
        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.resi_drop = nn.Dropout(0.1)
        self.c_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        if self.isMask:
            # self.register_buffer("mask", torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)))
                self.mask = torch.tril(torch.ones(cfg.pos_num, cfg.pos_num)).cuda()
    def forward(self, x):
        x = self.c_attn(x) # x形状(N,S,V)，N代表多少个句子，S代表多少个词，V代表每个词的维度
        x = x.reshape(*x.shape[:-1], cfg.head_num, -1)  # (N,S,V)——>(N,S,12,768/12*3)
        x = x.transpose(-2, -3)  # (N,S,12,768/12*3)——>(N,12,,S,768/12*3)
        q, k, v = x.chunk(3, dim=-1)
        w = (q @ k.transpose(-1, -2)) / self.dk  # (N,12,S,64)@(N,12,64,S)=(N,12,S,S)
        # if self.isMask:
        # mask=(self.mask if self.isMask else 1)
        mask=torch.tril(torch.ones(w.size(-2), w.size(-1))).cuda()
        w = w * mask - (1 - mask) * 1e5
        w = torch.softmax(w, dim=-1)
        w = self.attn_drop(w)
        a = w @ v  # (N,12,S,S)@(N,12,S,64)-->(N,12,S,64)
        a = a.transpose(-2, -3)  # (N,12,S,64)-->(N,S,12,64)
        a = a.reshape(*a.shape[:-2], cfg.embed_dim)  # (N,S,12,64)-->(N,S,768)
        h = self.c_proj(a)
        h = self.resi_drop(h)
        return h
class Block(nn.Module):
    def __init__(self, isMask=True):
        super().__init__()
        self.layer_normal_1 = nn.LayerNorm(cfg.embed_dim)
        self.attention = Attention(isMask)
        self.layer_normal_2 = nn.LayerNorm(cfg.embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.multi * cfg.embed_dim),
            nn.LeakyReLU(),
            nn.Linear(cfg.multi * cfg.embed_dim, cfg.embed_dim),
        )
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        h = self.layer_normal_1(x)
        a = self.attention(h)
        a = a + x  # 加一个残差
        a = self.layer_normal_2(a)
        h = self.proj(a)
        h = self.dropout(h)
        y = h + a  # 加一个残差
        return y
class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_embed = nn.Embedding(cfg.vocab_num, cfg.embed_dim) # 定义一个字典
        self.pos_embed = nn.Embedding(cfg.pos_num, cfg.embed_dim)   # 定义一个位置编码
        # self.type_embed = nn.Embedding(cfg.type_num, cfg.embed_dim)   # 定义一个类型编码
        self.blocks = []
        for _ in range(cfg.block_num):
            self.blocks.append(Block())
        self.drop = nn.Dropout(0.1)
        self.sequential = nn.Sequential(*self.blocks)
        self.output_layer = nn.Linear(cfg.embed_dim, cfg.vocab_num, bias=False)
    def forward(self, x, p):
        e = self.vocab_embed(x)  # 对输入进行词向量编码
        p = self.pos_embed(p)    # 对输入进行位置编码
        # t = self.type_embed(t)   # 对输入进行类型编码
        h = self.drop(e + p)
        h = self.sequential(h)
        return self.output_layer(h)
