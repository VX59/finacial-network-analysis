import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from itertools import combinations

days = 200
SAP = pd.read_csv(r"./archive/stocks/SAP.csv").tail(days)
FTNT = pd.read_csv(r"./archive/stocks/FTNT.csv").tail(days)
AMD = pd.read_csv(r"./archive/stocks/AMD.csv").tail(days)
OKTA = pd.read_csv(r"./archive/stocks/OKTA.csv").tail(days)
QCOM = pd.read_csv(r"./archive/stocks/QCOM.csv").tail(days)
ORCL = pd.read_csv(r"./archive/stocks/ORCL.csv").tail(days)
CSCO = pd.read_csv(r"./archive/stocks/CSCO.csv").tail(days)
ASML = pd.read_csv(r"./archive/stocks/ASML.csv").tail(days)
CHKP = pd.read_csv(r"./archive/stocks/CHKP.csv").tail(days)
ADSK = pd.read_csv(r"./archive/stocks/ADSK.csv").tail(days)
WDC = pd.read_csv(r"./archive/stocks/WDC.csv").tail(days)
MU = pd.read_csv(r"./archive/stocks/MU.csv").tail(days)
VST = pd.read_csv(r"./archive/stocks/VST.csv").tail(days)
TSLA = pd.read_csv(r"./archive/stocks/TSLA.csv").tail(days)
AVGO = pd.read_csv(r"./archive/stocks/AVGO.csv").tail(days)
VZ = pd.read_csv(r"./archive/stocks/VZ.csv").tail(days)
MDT = pd.read_csv(r"./archive/stocks/MDT.csv").tail(days)
CRWD = pd.read_csv(r"./archive/stocks/CRWD.csv").tail(days)
GOOGL = pd.read_csv(r"./archive/stocks/GOOGL.csv").tail(days)
CYBR = pd.read_csv(r"./archive/stocks/CYBR.csv").tail(days)
IBM = pd.read_csv(r"./archive/stocks/IBM.csv").tail(days)
GEN = pd.read_csv(r"./archive/stocks/GEN.csv").tail(days)
SHOP = pd.read_csv(r"./archive/stocks/SHOP.csv").tail(days)
TSCO = pd.read_csv(r"./archive/stocks/TSCO.csv").tail(days)
UBER = pd.read_csv(r"./archive/stocks/UBER.csv").tail(days)
ACM = pd.read_csv(r"./archive/stocks/ACM.csv").tail(days)
AAPL = pd.read_csv(r"./archive/stocks/AAPL.csv").tail(days)
ADBE = pd.read_csv(r"./archive/stocks/ADBE.csv").tail(days)
GRN = pd.read_csv(r"./archive/stocks/GRN.csv").tail(days)
IDE = pd.read_csv(r"./archive/stocks/IDE.csv").tail(days)
IRMD = pd.read_csv(r"./archive/stocks/IRMD.csv").tail(days)
RPD = pd.read_csv(r"./archive/stocks/RPD.csv").tail(days)

open_prices = {}
open_prices["SAP"] = SAP["Open"]
open_prices["FTNT"] = FTNT["Open"]
open_prices["AMD"] = AMD["Open"]
open_prices["OKTA"] = OKTA["Open"]
open_prices["QCOM"] = QCOM["Open"]
open_prices["ORCL"] = ORCL["Open"]
open_prices["CSCO"] = CSCO["Open"]
open_prices["ASML"] = ASML["Open"]
open_prices["CHKP"] = CHKP["Open"]
open_prices["ADSK"] = ADSK["Open"]
open_prices["WDC"] = WDC["Open"]
open_prices["MU"] = MU["Open"]
open_prices["VST"] = VST["Open"]
open_prices["TSLA"] = TSLA["Open"]
open_prices["AVGO"] = AVGO["Open"]
open_prices["VZ"] = VZ["Open"]
open_prices["MDT"] = MDT["Open"]
open_prices["CRWD"] = CRWD["Open"]
open_prices["GOOGL"] = GOOGL["Open"]
open_prices["CYBR"] = CYBR["Open"]
open_prices["IBM"] = IBM["Open"]
open_prices["GEN"] = GEN["Open"]
open_prices["SHOP"] = SHOP["Open"]
open_prices["TSCO"] = TSCO["Open"]
open_prices["UBER"] = UBER["Open"]
open_prices["ACM"] = ACM["Open"]
open_prices["AAPL"] = AAPL["Open"]
open_prices["ADBE"] = ADBE["Open"]
open_prices["GRN"] = GRN["Open"]
open_prices["IDE"] = IDE["Open"]
open_prices["IRMD"] =IRMD["Open"]
open_prices["RPD"] = RPD["Open"]

key_enum = list(enumerate(open_prices))
key_enum[0]

n = len(open_prices)
pairs = list(combinations(range(n),2))

A = np.zeros(shape=(len(open_prices),len(open_prices)))

def pearson_correlation_time(X,Y):
    XYPCT = np.zeros(1)

    for i in range(1,len(X)):
        cov = np.cov(X[:i],Y[:i], bias=True)[0,1]
        denominator = (np.std(X[:i])*np.std(Y[:i]))
        if(denominator != 0):
            PC = cov / denominator
        else:
            PC = 0
        XYPCT = np.append(XYPCT,PC)
    
    return XYPCT

import math

# construct MST to reveal underlying market relationships
def dist(corr):
    return [math.sqrt(2*(1-c)) for c in corr]

def pearson_correlation_window(X,Y):
    XYPCT = []
    window = 60
    for i in range(len(X)-window):
        cov = np.cov(X[i:window+i],Y[i:window+i], bias=True)[0,1]
        denominator = (np.std(X[i:window+i])*np.std(Y[i:window+i]))
        if(denominator != 0):
            PC = cov / denominator
        else:
            PC = 0
        XYPCT = np.append(XYPCT,PC)
        
    return XYPCT

pairwise_correlations = []

for p in pairs:
    pairwise_correlations.append(pearson_correlation_window(open_prices[key_enum[p[0]][1]],open_prices[key_enum[p[1]][1]]))

# BC and AC are not very highly correlated
mean = np.mean(pairwise_correlations,axis=0)
domain = list(range(len(mean)))

mean_expected_value = np.dot(mean, domain)
print("median expected value:", mean_expected_value)

def eval_pair_correlation(x,y, A):
    X = open_prices[key_enum[x][1]]
    Y = open_prices[key_enum[y][1]]
    XYPCT = pearson_correlation_window(X,Y)
    expected_value = np.dot(XYPCT,domain)
    if expected_value > mean_expected_value:
        A[x][y] = expected_value
        A[y][x] = expected_value

for p in pairs:
    eval_pair_correlation(p[0],p[1],A)

# creating the graph is O(n^2)
# lets to some spectral analysis

# L = D-A
ks = np.zeros(len(open_prices))
for k in range(len(open_prices)):
    for j in range(len(open_prices)):
        ks[k] += A[k][j]


D = np.diag(ks)
L = np.array(D - A)
Uv,U = np.linalg.eig(L)

def kmeans(X, k, max_iters=100, tol=1e-4):

    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids
    
    return labels

K=5
labels = kmeans(U[:,1:K+1], K)

unique_labels = np.unique(labels)
color_map = plt.cm.get_cmap("Set1", len(unique_labels))
colors = [color_map(label) for label in labels]
G = nx.from_numpy_array(A)

plt.figure(figsize=(20,20))

labels = {i: key_enum[i][1] for i in range(len(open_prices))}
pos = nx.spring_layout(G)
nx.draw(G, labels=labels, with_labels=True, node_color=colors, edge_color='gray', node_size=1000, font_size=16)
plt.savefig("portfolio_spectral_clustering.png")

#plt.show()

# now lets identify the most influential items
# Eigenvector centrality
AUv,AU = np.linalg.eig(A)

principle_eigv = AU[np.argmax(AUv)]
centrality = np.abs(principle_eigv)  # Taking absolute values to avoid negative centrality
centrality /= np.max(centrality)  # Normalize to range [0, 1]

G = nx.from_numpy_array(A)
node_colors = plt.cm.viridis(centrality)
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))

nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color=node_colors, font_size=12, edge_color='gray')

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=np.min(principle_eigv), vmax=np.max(centrality)))
sm.set_array([])
plt.colorbar(sm, ax=plt.gca(), label='Eigenvector Centrality')

plt.legend()
plt.savefig("portfolio_eigvec_centrality.png")
plt.show()
 
# find the MST Prim's algorithm

import sys

mst_set = [False]*len(A)

def min_key():
    min = sys.maxsize
    min_index = None

    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] != 0 and A[i][j] < min and not mst_set[i]:
                min = A[i][j]
                min_index = i

    return min_index

print(min_key())