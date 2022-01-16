import numpy as np
import pandas as pd
import os

data = pd.read_csv(os.path.join('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data.csv'), sep=',')

df2 = data[data.index % 2 == 0]  # Selects every 4th raw starting from 0

print(df2.shape[0])

df2.to_csv('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data_25.csv', index=False)

df2 = data[data.index % 4 == 0]  # Selects every 4th raw starting from 0

print(df2.shape[0])

df2.to_csv('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data_12.csv', index=False)

df2 = data[data.index % 8 == 0]  # Selects every 4th raw starting from 0

print(df2.shape[0])

df2.to_csv('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data_6.csv', index=False)

df2 = data[data.index % 16 == 0]  # Selects every 4th raw starting from 0

print(df2.shape[0])

df2.to_csv('C:/Users/karti/PycharmProjects/HAR/data/rwhar_data_3.csv', index=False)
