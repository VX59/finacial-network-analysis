import numpy as np
import pandas as pd
import os
import random

stocks_item_map = {}   #   (name, pd frame)
stocks_path = r"archive/stocks/"
n = 250

def map_market_item(csv_name:str, market_map:dict,  path):
    csv_path = path + csv_name
    item_name = csv_name[:-4]

    market_days = 400
    csv_data = pd.read_csv(csv_path)
    length = len(csv_data)

    if length >= market_days:
        print(f"length {length}")
        market_map[item_name] = csv_data.tail(market_days)["Adj Close"]

if (os.path.isdir(stocks_path)):
    csv_files = os.listdir(stocks_path)
    print(f"there are {len(csv_files)} items")
    map_market_item_vecfunc = np.vectorize(map_market_item)

    # take random slice
    random_samples = random.sample(csv_files, n)

    map_market_item_vecfunc(random_samples, stocks_item_map, stocks_path)

else:
    print("no stocks path")

np.savez("stock_item_map_small.npz", **stocks_item_map)
print(f"saved {len(stocks_item_map)} items to disk")