import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed

open_prices = dict(np.load("stock_item_map_small.npz", allow_pickle=True))

key_enum = list(enumerate(open_prices))
key_enum[0]

n = len(open_prices)
pairs = list(combinations(range(n),2))

pairwise_correlations = []

def pearson_correlation_window(i,j):
    X = open_prices[key_enum[i][1]]
    Y = open_prices[key_enum[j][1]]
    XYPCT = []
    window = 7
    for l in range(len(X)-window):
        cov = np.cov(X[l:window+l],Y[l:window+l], bias=True)[0,1]
        denominator = (np.std(X[l:window+l])*np.std(Y[l:window+l]))
        if(denominator != 0):
            PC = cov / denominator
        else:
            PC = 0
        XYPCT = np.append(XYPCT,PC)

    XYPCT = np.array(XYPCT)

    mean = np.nanmean(XYPCT)
    return XYPCT, mean, (i,j)


import time

start_time = time.perf_counter()
results = Parallel(n_jobs=-1, backend='loky')(delayed(pearson_correlation_window)(i, j) for i, j in pairs)
end_time = time.perf_counter()

time_delta = end_time - start_time
print(f"time delta {time_delta}")

XYPCTs, means, index_pairs = zip(*results)

threashold = np.percentile(means, 50)
print(f"threshold value {threashold}")

A = np.zeros(shape=(len(open_prices),len(open_prices)))

edges = 0
for _, mean, (i, j) in zip(XYPCTs, means, index_pairs):
    if mean >= threashold:
        edges += 1
        A[i][j] = mean
        A[j][i] = mean

print(f"tracked {edges} edges")
np.save("stock_item_adj_matrix_small_weighted.npy", A)