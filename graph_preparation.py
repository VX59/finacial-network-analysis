import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
import os

adj_close_prices = dict(np.load("stock_item_map_small.npz", allow_pickle=True))

key_enum = list(enumerate(adj_close_prices))
key_enum[0]

n = len(adj_close_prices)
pairs = list(combinations(range(n),2))

pairwise_correlations = []

sample_length = len(adj_close_prices[key_enum[0][1]])

window = 30
step_size = 100

os.makedirs("snapshots", exist_ok=True)

def pearson_correlation_window(i,j,t):
    X = adj_close_prices[key_enum[i][1]][t:t+window]
    Y = adj_close_prices[key_enum[j][1]][t:t+window]
    if len(X) < window or len(Y) < window:
        return 0, (i, j)
    
    cov = np.cov(X, Y, bias=True)[0, 1]
    denom = np.std(X) * np.std(Y)
    PC = cov / denom if denom != 0 else 0

    return PC, (i,j)

def correlation_snapshot(t):
    results = Parallel(n_jobs=-1, backend='loky')(delayed(pearson_correlation_window)(i, j, t) for i, j in pairs)

    PCs, index_pairs = zip(*results)

    A = np.zeros(shape=(len(adj_close_prices),len(adj_close_prices)))
    Aw = A.copy()
    for pc, (i, j) in zip(PCs, index_pairs):
        if pc >= 0.4:
            Aw[i][j] = Aw[j][i] = pc
            A[i][j] = A[j][i] = 1

    np.save(f"snapshots/{t/100}_A.npy", A)
    np.save(f"snapshots/{t/100}_Aw.npy", Aw)
    return t, np.count_nonzero(A) // 2

snapshots = []
for t in tqdm(range(0, sample_length - window, step_size)):
    snapshot = correlation_snapshot(t)
    snapshots.append(snapshot)

print("Completed snapshots:", snapshots)