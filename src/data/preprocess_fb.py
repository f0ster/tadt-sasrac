#!/usr/bin/env python3
"""Generate random‑walk sequences + train/test edge split for FB circles."""
import argparse, random, networkx as nx, numpy as np, tqdm, pathlib

def random_walk(G, start, length):
    walk=[start]
    for _ in range(length-1):
        cur=walk[-1]
        nbrs=list(G[cur]) or [cur]
        walk.append(random.choice(nbrs))
    return walk

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="data/raw/facebook_combined.txt")
    p.add_argument("--walks-per-node", type=int, default=10)
    p.add_argument("--walk-length", type=int, default=40)
    p.add_argument("--min-seq-len", type=int, default=5)
    p.add_argument("--out", default="data/processed/fb_walks.npz")
    args=p.parse_args()

    G=nx.read_edgelist(args.inp, nodetype=int)
    nodes=list(G.nodes())

    # random walks
    walks=[]
    for n in tqdm.tqdm(nodes):
        for _ in range(args.walks_per_node):
            walks.append(random_walk(G,n,args.walk_length))

    # 80/20 edge split
    edges=list(G.edges())
    random.shuffle(edges)
    split=int(0.8*len(edges))
    train_edges, test_edges = edges[:split], edges[split:]

    np.savez(args.out,
             walks=walks,
             train=train_edges,
             test=test_edges,
             n_nodes=len(nodes))
    print("Saved", args.out)