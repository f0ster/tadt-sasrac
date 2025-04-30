import torch, torch.nn as nn
from .sasrec import SASRec
from .mrtb import MRTB
from .tarc import TARC

class TADTRec(nn.Module):
    def __init__(self, n_items, d=128, K=6, max_len=40):
        super().__init__()
        self.item_emb=nn.Embedding(n_items, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.mrtbs   = nn.ModuleList([MRTB(d) for _ in range(K)])
        self.ctrl    = TARC(d, K*4)
        self.ln      = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, n_items)  # Project to vocabulary size
        self.K=K
    def forward(self, seq, seqlen):
        pos=torch.arange(seq.size(1), device=seq.device)
        h=self.item_emb(seq)+self.pos_emb(pos)
        route,beta=self.ctrl(h[:,-1], seqlen)
        for i,blk in enumerate(self.mrtbs):
            r=route[:,4*i:4*(i+1)]
            h=blk(h,r)
        h = self.ln(h[:,-1])
        logits = self.out_proj(h)  # Project to vocabulary size
        return logits, beta