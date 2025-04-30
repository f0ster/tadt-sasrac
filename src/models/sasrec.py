import torch, torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, n_items, d=128, n_layers=2, n_heads=2, max_len=40):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, d)
        self.pos_emb  = nn.Embedding(max_len, d)
        self.blocks   = nn.ModuleList([
            nn.TransformerEncoderLayer(d, n_heads, d*4, batch_first=True)
            for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        self.max_len=max_len

    def forward(self, seq):
        B,L = seq.shape
        pos = torch.arange(L, device=seq.device)
        h=self.item_emb(seq)+self.pos_emb(pos)
        for blk in self.blocks: h=blk(h)
        return self.ln(h)