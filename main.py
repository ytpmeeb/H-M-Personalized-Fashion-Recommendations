import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from warnings import resetwarnings


# https://www.kaggle.com/cdeotte/recommend-items-purchased-together-0-021/notebook
class purchasesHistory:
    def __init__(self, df):
        self.data = df
        self.train = []

    # find each customer's last week of purchases
    def lastWeekCustomerPurchases(self):
        self.data['t_dat'] = pd.to_datetime(self.data['t_dat'])
        tmp = self.data.groupby('customer_id').t_dat.max().reset_index()
        tmp.columns = ['customer_id', 'max_dat']
        self.train = self.data.merge(tmp, on=['customer_id'], how='left')
        self.train['diff_dat'] = (self.train.max_dat - self.train.t_dat).dt.days
        self.train = self.train.loc[self.train['diff_dat'] <= 6]

        return self.train

    # find top 50 most often purchased items in last week
    def lastWeekPopularPurchases(self):
        tmp = self.train.groupby(['customer_id', 'article_id'])['t_dat'].agg('count').reset_index()
        tmp.columns = ['customer_id', 'article_id', 'ct']
        self.train = self.train.merge(tmp, on=['customer_id', 'article_id'], how='left')
        self.train = self.train.sort_values(['ct', 't_dat'], ascending=False)
        self.train = self.train.drop_duplicates(['customer_id', 'article_id'])
        self.train = self.train.sort_values(['ct', 't_dat'], ascending=False)
        popularList = self.train[['article_id', 'ct']]

        return popularList[:50]



data = pd.read_csv('/Users/nakreond/Programming_Languages/Python_Projects/H&M Personalized Fashion Recommendations/H_M_transaction_2020.csv')
# change the time to sec
data['t_dat'] = pd.to_datetime(data['t_dat'], format='%Y-%m-%d').map(pd.Timestamp.timestamp)

purchasesHistory(data).lastWeekPurchases()

pd.crosstab(data.customer_id, data.article_id)
