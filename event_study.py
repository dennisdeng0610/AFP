# coding=utf-8
"""
@author : dennisdeng
@file   : afp_101521.py
@ide    : PyCharm
@time   : 10-15-2021 12:05:00
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:/Users/yanis/Desktop/Current Work/afp/reference/data')
stock_daily = pd.read_csv('Stock_data_daily.csv')
patent_sep = pd.read_csv('patent_data_separate.csv')

stock_daily = stock_daily[['Ticker', 'Date', 'Ret']]
stock_daily['Date'] = pd.to_datetime(stock_daily['Date'], format='%Y-%m-%d', errors='ignore')
patent_sep = patent_sep[['stock code', 'Patent Type', 'Application date']]
patent_sep['Application date'] = pd.to_datetime(patent_sep['Application date'], format='%Y-%m-%d', errors='ignore')
patent_sep = patent_sep.rename(columns={"stock code": "Ticker", 'Application date': 'Date'})
bound_year = True
if bound_year:
    patent_sep = patent_sep.loc[(patent_sep['Date'].dt.year <= 2005) & (patent_sep['Date'].dt.year >= 1995)]
# event-study on patent data - 5 trading days
stock_daily_1 = stock_daily.copy()
for i in range(1, 6):
    stock_daily_1['ret_f' + str(i)] = stock_daily_1['Ret'].shift(i)
for i in range(1, 6):
    stock_daily_1['ret_b' + str(i)] = stock_daily_1['Ret'].shift(-i)
for i in range(1, 6):
    stock_daily_1.loc[stock_daily_1['Ticker'].shift(i) != stock_daily_1['Ticker'], 'Date'] = np.nan
    stock_daily_1.loc[stock_daily_1['Ticker'].shift(-i) != stock_daily_1['Ticker'], 'Date'] = np.nan
stock_daily_1 = stock_daily_1[stock_daily_1['Date'].notna()]
stock_daily_1['diff'] = stock_daily_1.iloc[:, 8:13].mean(axis=1) - stock_daily_1.iloc[:, 3:8].mean(axis=1)
i = patent_sep.loc[patent_sep['Patent Type'] == 'i']
u = patent_sep.loc[patent_sep['Patent Type'] == 'u']
d = patent_sep.loc[patent_sep['Patent Type'] == 'd']
patent_stock = pd.merge(patent_sep, stock_daily_1, how='left', on=['Ticker', 'Date'])
i_stock = pd.merge(i, stock_daily_1, how='left', on=['Ticker', 'Date'])
u_stock = pd.merge(u, stock_daily_1, how='left', on=['Ticker', 'Date'])
d_stock = pd.merge(d, stock_daily_1, how='left', on=['Ticker', 'Date'])
patent_stock = patent_stock[patent_stock['Ret'].notna()]
i_stock = i_stock[i_stock['Ret'].notna()]
u_stock = u_stock[u_stock['Ret'].notna()]
d_stock = d_stock[d_stock['Ret'].notna()]

output_1 = pd.DataFrame(index=np.arange(4), columns=np.arange(2))
output_1.iloc[[0], [0]] = patent_stock['diff'].mean() * 252
output_1.iloc[[1], [0]] = i_stock['diff'].mean() * 252
output_1.iloc[[2], [0]] = u_stock['diff'].mean() * 252
output_1.iloc[[3], [0]] = d_stock['diff'].mean() * 252
output_1.iloc[[0], [1]] = patent_stock['diff'].mean() * np.sqrt(len(patent_stock)) / patent_stock['diff'].std()
output_1.iloc[[1], [1]] = i_stock['diff'].mean() * np.sqrt(len(i_stock)) / i_stock['diff'].std()
output_1.iloc[[2], [1]] = u_stock['diff'].mean() * np.sqrt(len(u_stock)) / u_stock['diff'].std()
output_1.iloc[[3], [1]] = d_stock['diff'].mean() * np.sqrt(len(d_stock)) / d_stock['diff'].std()
output_1.rename(
    columns={0: 'mean', 1: 't-stat'},
    index={0: 'overall', 1: 'invention', 2: 'utility', 3: 'design'},
    inplace=True)

# event-study on patent data - 10 natural days
pat_ticker = patent_sep['Ticker'].unique()
stock_daily_2 = stock_daily.loc[stock_daily['Ticker'].isin(pat_ticker)]
time_range = pd.date_range('1990-01-01 00:00:00', '2011-12-31 00:00:00', freq='d')
for i in range(0, len(pat_ticker)):
    data = stock_daily[stock_daily['Ticker'].isin([pat_ticker[i]])]
    data.set_index(['Date'], inplace=True)
    data = data.reindex(time_range)
    data.reset_index(drop=False, inplace=True)
    data = data.rename(columns={'index': 'Date'})
    data['Ticker'] = pat_ticker[i]
    if i == 0:
        output = data
    else:
        output = output.append(data)

stock_daily_2 = output
#
for i in range(1, 11):
    stock_daily_2['retf_' + str(i)] = stock_daily_2['Ret'].shift(i)
for i in range(1, 11):
    stock_daily_2['retb_' + str(i)] = stock_daily_2['Ret'].shift(-i)
for i in range(1, 11):
    stock_daily_2.loc[stock_daily_2['Ticker'].shift(i) != stock_daily_2['Ticker'], 'Date'] = np.nan
    stock_daily_2.loc[stock_daily_2['Ticker'].shift(-i) != stock_daily_2['Ticker'], 'Date'] = np.nan
stock_daily_2 = stock_daily_2[stock_daily_2['Date'].notna()]
stock_daily_2['diff'] = stock_daily_2.iloc[:, 13:23].mean(axis=1) - stock_daily_2.iloc[:, 3:13].mean(axis=1)
i = patent_sep.loc[patent_sep['Patent Type'] == 'i']
u = patent_sep.loc[patent_sep['Patent Type'] == 'u']
d = patent_sep.loc[patent_sep['Patent Type'] == 'd']
patent_stock = pd.merge(patent_sep, stock_daily_2, how='left', on=['Ticker', 'Date'])
i_stock = pd.merge(i, stock_daily_2, how='left', on=['Ticker', 'Date'])
u_stock = pd.merge(u, stock_daily_2, how='left', on=['Ticker', 'Date'])
d_stock = pd.merge(d, stock_daily_2, how='left', on=['Ticker', 'Date'])
patent_stock = patent_stock[patent_stock['diff'].notna()]
i_stock = i_stock.loc[i_stock['diff'].notna()]
u_stock = u_stock.loc[u_stock['diff'].notna()]
d_stock = d_stock.loc[d_stock['diff'].notna()]
output_2 = pd.DataFrame(index=np.arange(4), columns=np.arange(2))
output_2.iloc[[0], [0]] = patent_stock['diff'].mean() * 252
output_2.iloc[[1], [0]] = i_stock['diff'].mean() * 252
output_2.iloc[[2], [0]] = u_stock['diff'].mean() * 252
output_2.iloc[[3], [0]] = d_stock['diff'].mean() * 252
output_2.iloc[[0], [1]] = patent_stock['diff'].mean() * np.sqrt(len(patent_stock)) / patent_stock['diff'].std()
output_2.iloc[[1], [1]] = i_stock['diff'].mean() * np.sqrt(len(i_stock)) / i_stock['diff'].std()
output_2.iloc[[2], [1]] = u_stock['diff'].mean() * np.sqrt(len(u_stock)) / u_stock['diff'].std()
output_2.iloc[[3], [1]] = d_stock['diff'].mean() * np.sqrt(len(d_stock)) / d_stock['diff'].std()
output_2.rename(
    columns={0: 'mean', 1: 't-stat'},
    index={0: 'overall', 1: 'invention', 2: 'utility', 3: 'design'},
    inplace=True)
