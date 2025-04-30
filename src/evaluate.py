import numpy as np, torch, tqdm, argparse
from models.tadt_rec import TADTRec
from models.sasrec import SASRec

parser=argparse.ArgumentParser()
parser.add_argument("--ckpt"); zparser.add_argument("--tadt",action="store_true")
args=parser.parse_args()

data=np.load("data/processed/fb_walks.npz", allow_pickle=True)
train_edges=data["train"]; test_edges=data["test"]

n_items=int(data["n_nodes"])
model=(TADTRec if args.tadt else SASRec)(n_items)
state=torch.load(args.ckpt, map_location="cpu"); model.load_state_dict(state)
model.eval()
emb=model.item_emb.weight.data

hits=0
for u,v in tqdm.tqdm(test_edges):
    scores=torch.mv(emb, emb[u])
    topk=scores.topk(11).indices  # include u itself
    if v in topk: hits+=1
print("Hit@10:", hits/len(test_edges))