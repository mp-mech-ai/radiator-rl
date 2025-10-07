# Download historical data: https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/puy/ogd-smn_puy_t_historical_2020-2029.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

filename = "data/raw/ogd-smn_puy_t_historical_2020-2029.csv"

df = pd.read_csv(filename, sep=";")

df = df[df.columns[1:3]]
df.columns = ["Date", "T_out"]
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y %H:%M")

df = df[df["Date"] > datetime(year=2024, month=1, day=1)]
df = df.set_index("Date")
print(df.info)

print(df.describe())
df.to_csv("data/clean/t_out.csv")