import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
# A B and C are items in a market with value that changes over time. we could consider them random variables that are slightly dependent on one another
# we want to see how much the items "pull on eachother" by obvserving the behavior of their corelation as time advances
feature_pairs = 3*(3-1)/2

A = [1,2,3,3,6,7,3,4,4,3,2,2,3,4,5]
B = [0,0,1,0,5,5,3,2,3,3,1,1,4,5,2]
C = [0,1,1,1,2,2,3,4,5,3,1,1,2,1,2]

AM = np.zeros(shape=(feature_pairs,feature_pairs))

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

pairwise_correlations = []

pairwise_correlations.append(pearson_correlation_time(A,B))
pairwise_correlations.append(pearson_correlation_time(B,C))
pairwise_correlations.append(pearson_correlation_time(A,C))

print(pairwise_correlations)
# BC and AC are not very highly correlated
median = np.median(pairwise_correlations,axis=0)

print(median)

x = list(range(len(A)))

median_expected_value = np.dot(median, x)

# each edge is for a unique combination how do we decode i to the appropriate nodes
print(median_expected_value)

for i,p in enumerate(pairwise_correlations):
    expected_value = np.dot(p,x)
    if expected_value > median_expected_value:
        print(i, expected_value)

plt.plot(x,A)
plt.plot(x,B)
plt.plot(x,C)
plt.show()
# we should draw an edge between A and B in our graph based on the market conditions
AM[1][0] = 1
AM[0][1] = 1

G = nx.from_numpy_array(AM)

plt.figure(figsize=(6,6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)
plt.show()


# next scale this example up using real market data
# no 2 items can assumed to have been on the market for the same amount of time so we cant really expect to
# have our data go back many years, maybe we can evaluate 1 year or 1 month windows
# we would want to be able to predict some market trends based on indicators. such as 2 groups are strongly connected
# if one of those groups suffers then the other one will follow etc. this requires some level of spectral analysis and is very
# computationally expensive