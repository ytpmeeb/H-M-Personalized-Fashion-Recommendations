"""
03/21 Meeting
1. In the presentation, we can discuss the failre of using K-mean

2. Two parts of data cleaning need to complete:
    (1) Reduce the category type of product to less than 1000
    (2) Find the top 100 or top 50 most popular product in each year
        ***the reason don't find the next day purchase is because the next transaction may be highly related with
        the previous one and that fact is a command sense, ex: When A bought a new phone, high chance to buy the
        phone case next day
    (3) Find customer who made more than 2 transaction in the transaction history

2. New models to focus are:
    (1) Pattern (unsupervised learning): Apriori
    (2) Classification: Logistic regression
        i. the table should focus on one category at a time, ex: who buy t-shirt will they buy shoe in next 7 days?
        column: t-shirt(Y/N) | Online or Instore | buy shoe in next 7 days(Y/N)
        row: the events (the frequency of events )
             | t-shirt  |  Location  |   buy shoe in next 7 days
        ----------------------------------------------------------
        E1  |    Y     |    Online  |               Y

3. Compare or use different method to get the answer


**aggression detection machine learning

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
    @staticmethod
    # find each customer's last week of purchases
    def lastWeekCustomerPurchases(data):
        train = []
        data['t_dat'] = pd.to_datetime(data['t_dat'])
        tmp = data.groupby('customer_id').t_dat.max().reset_index()
        tmp.columns = ['customer_id', 'max_dat']
        train = data.merge(tmp, on=['customer_id'], how='left')
        train['diff_dat'] = (train.max_dat - train.t_dat).dt.days
        train = train.loc[train['diff_dat'] <= 6]

        return train

    @staticmethod
    # find top 50 most often purchased items in last week
    def lastWeekPopularPurchases(train):
        tmp = train.groupby(['customer_id', 'article_id'])['t_dat'].agg('count').reset_index()
        tmp.columns = ['customer_id', 'article_id', 'ct']
        train = train.merge(tmp, on=['customer_id', 'article_id'], how='left')
        train = train.sort_values(['ct', 't_dat'], ascending=False)
        train = train.drop_duplicates(['customer_id', 'article_id'])
        train = train.sort_values(['ct', 't_dat'], ascending=False)
        popularList = train[['article_id', 'ct']]

        return popularList[:50]


data = pd.read_csv('./Data/Training Data/transaction_2020.csv')
data = pd.read_csv('./Data/Training Data/training_customer.csv')
data = pd.read_csv('./Data/Training Data/training_article.csv')
# change the time to sec
data['t_dat'] = pd.to_datetime(data['t_dat'], format='%Y-%m-%d').map(pd.Timestamp.timestamp)
# skip the id
data = data.iloc[:, 1:]

m_nor = Data_Transformation.min_max(data)
z_nor = Data_Transformation.zScore(data)

m_pca = Data_Reduction.pca(m_nor)
z_pca = Data_Reduction.pca(z_nor)

Unsupervised_Learning.k_mean_find_k(m_nor)
Unsupervised_Learning.k_mean_find_k(z_nor)

m_k = Unsupervised_Learning.k_mean(m_pca, 10)
z_k = Unsupervised_Learning.k_mean(z_pca, 10)


PurchasesHistory(data).lastWeekCustomerPurchases()

# Transform data into 0 and 1
pd.crosstab(data.customer_id, data.article_id)
pd.crosstab(data.customer_id, data.t_dat)
pd.crosstab(data.article_id, data.t_dat)

