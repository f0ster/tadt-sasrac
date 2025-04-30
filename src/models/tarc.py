import torch.nn as nn, torch

class TARC(nn.Module):
    """Task-Aware Routing Controller for dynamic path selection."""
    def __init__(self, d, K):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d, 4*K),
            nn.Sigmoid()
        )
    def forward(self, x, seqlen):
        route = self.ffn(x)
        beta = route.mean()
        return route, beta