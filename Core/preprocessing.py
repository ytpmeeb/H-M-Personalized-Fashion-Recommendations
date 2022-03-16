import pandas as pd
from datetime import *


# reduce memory by transforming customer_id from 64 bytes (String) to 8 bytes (Int)
# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
def reduceMemory(name, df):
    df[name] = train[name].apply(lambda x: int(x[-16:], 16)).astype('int64')
    return df


# Data Reduction on customer.csv
customer_csv = 'C:/Users/naerkond/Programming_Languages/Projects_Python/Playground/customer.csv'
train = pd.read_csv(customer_csv)
reduceMemory('customer_id', train)
train.to_csv('customer_new.csv', sep=',', encodings='UTF-8', index=None, header=True)


# Data Reduction on transaction.csv
tx_csv = 'C:/Users/naerkond/Programming_Languages/Projects_Python/Playground/transactions_train.csv'
train = pd.read_csv(tx_csv)
train['t_dat'] = pd.to_datetime(train['t_dat'])
train['t_dat'] = train['t_dat'].dt.date
reduceMemory('customer_id', train)

# split transaction into years and generate csv
tx2018 = train.loc[train['t_dat'] <= date(2019, 9, 20)]
tx2019 = train.loc[(date(2019, 9, 20) < train['t_dat']) & (train['t_dat'] <= date(2020, 9, 20))]
tx2020 = train.loc[(date(2020, 9, 20) < train['t_dat']) & (train['t_dat'] <= date(2021, 9, 20))]
tx2018.to_csv('transaction_2018.csv', sep=',', encoding='UTF-8', index=None, header=True)
tx2019.to_csv('transaction_2019.csv', sep=',', encoding='UTF-8', index=None, header=True)
tx2020.to_csv('transaction_2020.csv', sep=',', encoding='UTF-8', index=None, header=True)


