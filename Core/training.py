from Tools import Anomaly_Detection
from Tools import Data_Integration
from Tools import Data_Reduction
from Tools import Data_Transformation
from Tools import Supervised_Learning
from Tools import Unsupervised_Learning
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


class Clustering:
    @staticmethod
    def training_k_mean(data):
        # skip the ID
        mNor = Data_Transformation.min_max(data.iloc[:, 1:])
        zNor = Data_Transformation.z_score(data.iloc[:, 1:])

        mPca = Data_Reduction.pca(mNor)
        zPca = Data_Reduction.pca(zNor)

        Unsupervised_Learning.k_mean_find_k(mPca)
        Unsupervised_Learning.k_mean_find_k(zPca)

        mK = Unsupervised_Learning.k_mean(mPca, 10)
        zK = Unsupervised_Learning.k_mean(zPca, 10)


class PurchasesHistory:  # https://www.kaggle.com/cdeotte/recommend-items-purchased-together-0-021/notebook
    @staticmethod
    # find each customer's last week of purchases
    def lastWeekCustomerPurchases(data):
        train = []
        data['t_dat'] = pd.to_datetime(data['t_dat'])
        temp = data.groupby('customer_id').t_dat.max().reset_index()
        temp.columns = ['customer_id', 'max_dat']
        train = data.merge(temp, on=['customer_id'], how='left')
        train['diff_dat'] = (train.max_dat - train.t_dat).dt.days
        train = train.loc[train['diff_dat'] <= 6]

        return train

    @staticmethod
    # find top 50 most often purchased items in last week
    def lastWeekPopularPurchases(data):
        temp = data.groupby(['customer_id', 'article_id'])['t_dat'].agg('count').reset_index()
        temp.columns = ['customer_id', 'article_id', 'ct']
        train = data.merge(temp, on=['customer_id', 'article_id'], how='left')
        train = train.drop_duplicates(['customer_id', 'article_id'])
        train = train.sort_values(['ct', 't_dat'], ascending=False)
        popularList = train[['article_id', 'ct']]

        return popularList[:50]

