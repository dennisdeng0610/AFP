# -*- coding: utf-8 -*-
"""
Created on Sun May 16 00:05:42 2021

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from linearmodels.panel import PanelOLS, FamaMacBeth

os.chdir("C:/Users/DELL/Desktop/MFE Spring/AFP")

App_Data = pd.read_excel('Application Data.xlsx')

fig, ax = plt.subplots(figsize=(8, 6))
App_Data['firm year'].hist(bins = 20, label = 'Application')
App_Data.loc[App_Data['Grant'] == 1]['firm year'].hist(bins = 20, label = 'Grant')
ax.set_title('Petent Application Each Year (1990 - 2010)')
ax.set_xlabel('Year')
ax.set_ylabel('Number')
plt.legend()

# invention patents, utility model patents (similar to the German Gebrauchsmuster), and design patents
fig, ax = plt.subplots(figsize=(8, 6))
u = list(App_Data['Patent Type']).count('u')
i = list(App_Data['Patent Type']).count('i')
d = list(App_Data['Patent Type']).count('d')
patent = ['Invention patents','Utility model patents','Design patents']
num_patent = np.array([i, u, d])
num_patent/sum(num_patent)

ax.pie(num_patent, labels = patent)
ax.set_title('Petent Type (1990 - 2010)')

# More strict examination process
fig, ax = plt.subplots(figsize=(8, 6))
Grant_Data = App_Data[['firm year','Grant']].groupby('firm year')['Grant'].sum()/\
    App_Data[['firm year','Grant']].groupby('firm year')['Grant'].size()
Grant_Data.plot()
ax.set_title('Patent Grants Ratio (1990 - 2010)')
ax.set_xlabel('Year')
ax.set_ylabel('Ratio')

stock_data = pd.read_csv('stock_data.csv')
stock_data = stock_data[['date', 'ticker', 'return', 'market cap']].copy()
stock_data['log_tot_ret'] = np.log(stock_data['return']+1)
stock_data['date'] = pd.to_datetime(stock_data['date'])
stock_data['Year'] = stock_data['date'].dt.year
stock_data_annual = stock_data.groupby(['ticker', 'Year'])['log_tot_ret'].aggregate([np.sum,np.size])
# Compute the market caps
stock_data_annual['avg_mkt_cap'] = np.array(stock_data.groupby(['ticker', 'Year'])['market cap'].mean())
stock_data_annual['avg_mkt_cap'] = stock_data_annual['avg_mkt_cap']/1e6 # Measure in millions
# Compute the returns from log rets
stock_data_annual['ret'] = np.exp(stock_data_annual['sum'])-1
# Only use the observations with sample size larger than 12 month
stock_data_annual = stock_data_annual.loc[stock_data_annual['size']==12]
stock_data_annual = stock_data_annual.drop(columns = ['size','sum']).reset_index()
# Moving Avg returns on different horizons
stock_data_annual['ret_2'] = np.array(stock_data_annual.groupby(['ticker'])['ret'].rolling(2).mean().shift(-1))
stock_data_annual['ret_3'] = np.array(stock_data_annual.groupby(['ticker'])['ret'].rolling(3).mean().shift(-2))
stock_data_annual['ret_5'] = np.array(stock_data_annual.groupby(['ticker'])['ret'].rolling(5).mean().shift(-4))
stock_data_annual['ret_7'] = np.array(stock_data_annual.groupby(['ticker'])['ret'].rolling(7).mean().shift(-6))
stock_data_annual['ret_10'] = np.array(stock_data_annual.groupby(['ticker'])['ret'].rolling(10).mean().shift(-9))

# Patent data part
patent_data_annual = pd.read_excel('Annual Data.xlsx')
patent_data_annual['I'] = 1/3*(patent_data_annual['I1'] + patent_data_annual['I2'] + patent_data_annual['I3'])
patent_data_annual['I'] = np.where(patent_data_annual['I'] == 0,0.001,patent_data_annual['I'])
patent_data_annual['I_log'] = np.log(patent_data_annual['I'])

patent_data_annual['U'] = 1/3*(patent_data_annual['U1'] + patent_data_annual['U2'] + patent_data_annual['U3'])
patent_data_annual['U'] = np.where(patent_data_annual['U'] == 0,0.001,patent_data_annual['U'])
patent_data_annual['U_log'] = np.log(patent_data_annual['U'])

patent_data_annual['D'] = 1/3*(patent_data_annual['D1'] + patent_data_annual['D2'] + patent_data_annual['D3'])
patent_data_annual['D'] = np.where(patent_data_annual['D'] == 0,0.001,patent_data_annual['D'])
patent_data_annual['D_log'] = np.log(patent_data_annual['D'])

# Set up penal regression  
patent_data_annual['Year'] = patent_data_annual['year'] + 1
patent_data_annual_m = patent_data_annual[['Year','stock code', 'I', 'U', 'D']].copy()
patent_data_annual_m = patent_data_annual_m.rename(columns={'stock code':'ticker'})
patent_data_annual_m = patent_data_annual_m.merge(stock_data_annual, how = 'right', on = ['Year','ticker'])
patent_data_annual_m.sort_values(['ticker','Year'], ascending=[True, True], inplace=True)
patent_data_annual_m =  patent_data_annual_m.set_index(['ticker','Year'])
ind_na = patent_data_annual_m['I'].isna() & patent_data_annual_m['U'].isna() & patent_data_annual_m['D'].isna()
patent_data_annual_m = patent_data_annual_m.loc[~ind_na]

# Add interactive terms
patent_data_annual_m['I_mktCap'] = np.log(patent_data_annual_m['I'] * patent_data_annual_m['avg_mkt_cap'])
patent_data_annual_m['U_mktCap'] = np.log(patent_data_annual_m['U'] * patent_data_annual_m['avg_mkt_cap'])
patent_data_annual_m['D_mktCap'] = np.log(patent_data_annual_m['D'] * patent_data_annual_m['avg_mkt_cap'])

ret_panel_1 = PanelOLS.from_formula(formula = 'ret ~ 1 + I + U + D + I_mktCap + U_mktCap + D_mktCap + TimeEffects',
                                    data=patent_data_annual_m).fit(cov_type = 'clustered', cluster_entity=True, cluster_time=True)
ret_panel_2 = PanelOLS.from_formula(formula = 'ret_2 ~ 1 + I + U + D + I_mktCap + U_mktCap + D_mktCap + TimeEffects',
                                    data=patent_data_annual_m).fit(cov_type = 'clustered', cluster_entity=True, cluster_time=True)
ret_panel_3 = PanelOLS.from_formula(formula = 'ret_3 ~ 1 + I + U + D + I_mktCap + U_mktCap + D_mktCap + TimeEffects',
                                    data=patent_data_annual_m).fit(cov_type = 'clustered', cluster_entity=True, cluster_time=True)
ret_panel_5 = PanelOLS.from_formula(formula = 'ret_5 ~ 1 + I + U + D + I_mktCap + U_mktCap + D_mktCap + TimeEffects',
                                    data=patent_data_annual_m).fit(cov_type = 'clustered', cluster_entity=True, cluster_time=True)
ret_panel_7 = PanelOLS.from_formula(formula = 'ret_7 ~ 1 + I + U + D + I_mktCap + U_mktCap + D_mktCap + TimeEffects',
                                    data=patent_data_annual_m).fit(cov_type = 'clustered', cluster_entity=True, cluster_time=True)
ret_panel_10 = PanelOLS.from_formula(formula = 'ret_10 ~ 1 + I + U + D + I_mktCap + U_mktCap + D_mktCap + TimeEffects',
                                    data=patent_data_annual_m).fit(cov_type = 'clustered', cluster_entity=True, cluster_time=True)

#patent_data_annual_m.to_pickle('Panel_Regression.pkl')

for i in range(1, 16, 1):
    patent_data_annual['Year'] = patent_data_annual['year'] + i
    
    patent_data_annual_m = patent_data_annual[['Year','stock code', 'I', 'U', 'D']].copy()
    patent_data_annual_m = patent_data_annual_m.rename(columns={'stock code':'ticker'})
    patent_data_annual_m = patent_data_annual_m.merge(stock_data_annual, how = 'left', on = ['Year','ticker'])
    patent_data_annual_m.sort_values(['ticker','Year'], ascending=[True, True], inplace=True)
    patent_data_annual_m =  patent_data_annual_m.set_index(['ticker','Year'])
    patent_data_annual_m.dropna(inplace = True)
    
    ret_panel = PanelOLS.from_formula(formula = 'ret ~ 1 + I + U + D + TimeEffects', data=patent_data_annual_m).fit(cov_type = 'clustered', 
                                          cluster_entity=True, cluster_time=True)
    print(i)
    print(ret_panel)
    

