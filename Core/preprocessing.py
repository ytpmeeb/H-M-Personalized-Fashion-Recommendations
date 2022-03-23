import pandas as pd
from datetime import *


# reduce memory by transforming customer_id from 64 bytes (String) to 8 bytes (Int)
# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
def reduceMemory(name, df):
    df[name] = df[name].apply(lambda x: int(x[-16:], 16)).astype('int64')
    return df


def customer_data(data: []):
    reduceMemory('customer_id', data)
    reduceMemory('postal_code', data)

    # find the unique value
    pd.unique(data['club_member_status'])

    # check and delete duplication
    if data.duplicated(subset=['customer_id']).sum() > 0:
        data.drop_duplicates(subset=['customer_id'])

    # check and delete missing value
    data = data.drop('FN', 1)
    data = data.drop('Active', 1)
    data = data.dropna(subset=['age', 'club_member_status', 'fashion_news_frequency'])

    # change club_member_status to numeric: ACTIVE=2, PRE-CREATE=1, LEFT CLUB=0
    data.loc[data['club_member_status'] == 'ACTIVE', 'club_member_status'] = 2
    data.loc[data['club_member_status'] == 'PRE-CREATE', 'club_member_status'] = 1
    data.loc[data['club_member_status'] == 'LEFT CLUB', 'club_member_status'] = 0

    # change fashion_news_frequency to numeric: Monthly=2, Regularly=1, NONE(None)=0
    data.loc[data['fashion_news_frequency'] == 'Monthly', 'fashion_news_frequency'] = 2
    data.loc[data['fashion_news_frequency'] == 'Regularly', 'fashion_news_frequency'] = 1
    data.loc[data['fashion_news_frequency'] == 'NONE', 'fashion_news_frequency'] = 0
    data.loc[data['fashion_news_frequency'] == 'None', 'fashion_news_frequency'] = 0

    # write to csv
    data.to_csv('training_customer.csv', sep=',', encoding='UTF-8', index=None, header=True)


def transaction_data(data: []):
    data['t_dat'] = pd.to_datetime(data['t_dat'])
    data['t_dat'] = data['t_dat'].dt.date
    reduceMemory('customer_id', data)

    # split transaction into years and generate csv
    tx2018 = data.loc[data['t_dat'] <= date(2019, 1, 3)]
    tx2019 = data.loc[(date(2019, 1, 3) < data['t_dat']) & (data['t_dat'] <= date(2020, 1, 2))]
    tx2020 = data.loc[(date(2020, 1, 2) < data['t_dat']) & (data['t_dat'] <= date(2021, 9, 17))]
    tx2020Last = data.loc[(date(2020, 9, 17) < data['t_dat'])]

    # find customer who purchase more than 1 times
    rawC = data.drop_duplicates(['customer_id', 't_dat'])
    trainC = rawC.groupby(['customer_id'])['article_id'].agg('count').reset_index()
    trainC = trainC.drop_duplicates(['customer_id'])
    trainC.columns = ['customer_id', 'ct']
    trainC = trainC.sort_values(['ct', 'customer_id'], ascending=False)
    trainC = trainC.loc[trainC['ct'] >= 2]

    # find product from multi-purchases customers
    rawA = data.loc[data['customer_id'].isin(trainC['customer_id'])]
    trainA = rawA.groupby(['customer_id', 'article_id'])['t_dat'].agg('count').reset_index()
    trainA.columns = ['customer_id', 'article_id', 'ct']
    trainA = data.merge(trainA, on=['customer_id', 'article_id'], how='left')
    trainA = trainA.drop_duplicates(['customer_id', 'article_id'])
    trainA = trainA.sort_values(['ct', 't_dat'], ascending=False)
    popularList = trainA[['article_id', 'ct']]
    popularList[:50]

    # write to csv
    trainC.to_csv('multi-purchases_customers.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2018.to_csv('training_transaction_2018.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2019.to_csv('training_transaction_2019.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2020.to_csv('training_transaction_2020.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2020Last.to_csv('training_transaction_2020_last_week.csv', sep=',', encoding='UTF-8', index=None, header=True)


def article_data(data: []):
    # delete nonnumerical  value
    data = data.drop('prod_name', 1)
    data = data.drop('product_type_name', 1)
    data = data.drop('product_group_name', 1)
    data = data.drop('graphical_appearance_name', 1)
    data = data.drop('colour_group_name', 1)
    data = data.drop('perceived_colour_value_name', 1)
    data = data.drop('perceived_colour_master_name', 1)
    data = data.drop('department_name', 1)
    data = data.drop('index_code', 1)
    data = data.drop('index_name', 1)
    data = data.drop('index_group_name', 1)
    data = data.drop('section_name', 1)
    data = data.drop('garment_group_name', 1)
    data = data.drop('detail_desc', 1)

    # check and delete missing value
    data = data.dropna()

    # write to csv
    data.to_csv('training_article.csv', sep=',', encoding='UTF-8', index=None, header=True)

