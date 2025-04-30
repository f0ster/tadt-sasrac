import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class WalkDataset(Dataset):
    def __init__(self, npz_path, window=5, max_len=40):
        d = np.load(npz_path, allow_pickle=True)
        self.walks = d["walks"]
        self.window = window
        self.max_len = max_len
        self.n_nodes = int(d["n_nodes"])

        # build (center, context, seqlen) triples ahead of time
        triples = []
        for w in self.walks:
            # Ensure all node IDs are valid
            w = np.clip(w, 0, self.n_nodes - 1)
            for i, u in enumerate(w):
                for j in range(max(0, i-window), min(len(w), i+window+1)):
                    if i == j:
                        continue
                    triples.append((int(u), int(w[j]), len(w)))
        self.data = triples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(batch):
    center, ctx, seqlen = zip(*batch)
    center = torch.tensor(center, dtype=torch.long)  # B
    ctx = torch.tensor(ctx, dtype=torch.long)
    seqlen = torch.tensor(seqlen, dtype=torch.long)
    return center, ctx, seqlen

def get_loader(npz, batch_size=512, workers=4):
    ds = WalkDataset(npz)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=True
    ), ds.n_nodes