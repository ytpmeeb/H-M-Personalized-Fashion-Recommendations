"""
https://www.youtube.com/watch?v=WGlMlS_Yydk
https://www.youtube.com/watch?v=SVM_pX0oTU8
https://www.geeksforgeeks.org/implementing-apriori-algorithm-in-python/
03/21 & 3/23 Meeting
1. In the presentation, we can discuss the failre of using K-mean

2. Two parts of data cleaning need to complete:
    (1) Reduce the category type of product to less than 1000
    (2) Find the top 100 or top 50 most popular product in each year
        ***the reason don't find the next day purchase is because the next transaction may be highly related with
        the previous one and that fact is a command sense, ex: When A bought a new phone, high chance to buy the
        phone case next day
    (3) Find customer who made more than 2 transaction in the transaction history (find the customer who puchase in next nweek   )

2. New models to focus are:
    (1) Pattern (unsupervised learning): FP Growth (this algorithm is faster than apriori)
        *** Apriori can set (support threshold to 60%, confidence to 40%)
    (2) Classification: Logistic regression
        i. the table should focus on one category at a time, ex: who buy t-shirt will they buy shoe in next 7 days?
        column: t-shirt(Y/N) | Online or Instore | buy shoe in next 7 days(Y/N)
        row: the events (the frequency of events )
             | t-shirt  |  Location  |   buy shoe in next 7 days
        ----------------------------------------------------------
        E1  |    Y     |    Online  |               Y

3. training:
    (1) train on 2019 and test on 2020
    (2) predict the odds of the same product been bought again
    (3) based one the recently record to sugeest the product
    (4)

4. suggesting list:


**aggression detection machine learning
--------------------------------------------------------------------------------------------------------------------------------

16 Meeting

focus on:
1. project should focus more on Event of item, like what item was bought at what day
row is time, range can be last 30 days purchase
col is purchase event, the feature what customer did or product

2. eliminate the one time shopper and focus on more than 2 times purchase customer

3. customer who have more than 2 times purchase vs who didn't purchase

4. what customer do before they purchase

5. use the pattern discover to find out first, then use supervise classification to find out correct answers


training dataset can be separated in tables:
1. user: profile about each user
2. purchase with  item and who buy it (time, customerID,  itemID)
3. what item belong to what category and price
"""

from Core import training
from Core import preprocessing
# from Core import exploring
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from datetime import *


rawT = pd.read_csv('./Data/Original Data/transactions_train.csv')
rawC = pd.read_csv('./Data/Original Data/customers.csv')
rawA = pd.read_csv('./Data/Original Data/articles.csv')
trainT = pd.read_csv('./Data/Training Data/training_transaction_2020_last_week.csv')
trainT18 = pd.read_csv('./Data/Training Data/training_transaction_2018.csv')
trainT19 = pd.read_csv('./Data/Training Data/training_transaction_2019.csv')
trainT20 = pd.read_csv('./Data/Training Data/training_transaction_2020.csv')
trainC = pd.read_csv('./Data/Training Data/training_customer.csv')
trainMC = pd.read_csv('./Data/Training Data/multi-purchases_customers.csv')
trainA = pd.read_csv('./Data/Training Data/training_article.csv')


# preprocessing
preprocessing.transaction_data(rawT)
preprocessing.customer_data(rawC)
preprocessing.article_data(rawA)


# training
# change the time to sec
trainT['t_dat'] = pd.to_datetime(trainT['t_dat'], format='%Y-%m-%d').map(pd.Timestamp.timestamp)

training.Clustering.training_k_mean(trainC)
training.Clustering.training_k_mean(trainA)

training.PurchasesHistory.lastWeekCustomerPurchases(trainT)

# Transform data into 0 and 1
pd.crosstab(trainT.customer_id, trainT.article_id)
pd.crosstab(trainT.customer_id, trainT.t_dat)
pd.crosstab(trainT.article_id, trainT.t_dat)


