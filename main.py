"""
03/16 Meeting

focus on:
1. project should focus more on Event of item, like what item was bought at what day
row is time, range can be last 30 days purchase
col is purchase event, the feature what customer did or product

2. eliminate the one time shopper and focus on more than 2 times purchase customer

3. customer who have more than 2 times purchase vs who didn't purchase

4. what customer do before they purchase

5. use the pattern discover to find out first, than  use supervise classification to find out correct answers


training dataset can be separated in tables:
1. user: profile about each user
2. purchase with ITEM item and who buy it (time, customerID,  itemID)
3. what item belong to what category and price
"""

from Tools import Anomaly_Detection
from Tools import Data_Integration
from Tools import Data_Reduction
from Tools import Data_Transformation
from Tools import Supervised_Learning
from Tools import Unsupervised_Learning
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import *


# https://www.kaggle.com/cdeotte/recommend-items-purchased-together-0-021/notebook
class PurchasesHistory:
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

purchasesHistory(data).lastWeekCustomerPurchases()

# Transform data into 0 and 1
pd.crosstab(data.customer_id, data.article_id)

