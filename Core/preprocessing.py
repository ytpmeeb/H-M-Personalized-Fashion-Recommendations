import pandas as pd
from datetime import *


# reduce memory by transforming customer_id from 64 bytes (String) to 8 bytes (Int)
# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
def reduceMemory(name, df):
    df[name] = df[name].apply(lambda x: int(x[-16:], 16)).astype('int64')
    return df


def customer(train: []):
    reduceMemory('customer_id', train)
    reduceMemory('postal_code', train)

    # find the unique value
    # pd.unique(test['club_member_status'])

    # check and delete duplication
    if train.duplicated(subset=['customer_id']).sum() > 0:
        train.drop_duplicates(subset=['customer_id'])

    # check and delete missing value
    train = train.drop('FN', 1)
    train = train.drop('Active', 1)
    train = train.dropna(subset=['age', 'club_member_status', 'fashion_news_frequency'])

    # change club_member_status to numeric: ACTIVE=2, PRE-CREATE=1, LEFT CLUB=0
    train.loc[train['club_member_status'] == 'ACTIVE', 'club_member_status'] = 2
    train.loc[train['club_member_status'] == 'PRE-CREATE', 'club_member_status'] = 1
    train.loc[train['club_member_status'] == 'LEFT CLUB', 'club_member_status'] = 0

    # change fashion_news_frequency to numeric: Monthly=2, Regularly=1, NONE(None)=0
    train.loc[train['fashion_news_frequency'] == 'Monthly', 'fashion_news_frequency'] = 2
    train.loc[train['fashion_news_frequency'] == 'Regularly', 'fashion_news_frequency'] = 1
    train.loc[train['fashion_news_frequency'] == 'NONE', 'fashion_news_frequency'] = 0
    train.loc[train['fashion_news_frequency'] == 'None', 'fashion_news_frequency'] = 0

    train.to_csv('training_customer.csv', sep=',', encoding='UTF-8', index=None, header=True)


def transaction(train: []):
    train['t_dat'] = pd.to_datetime(train['t_dat'])
    train['t_dat'] = train['t_dat'].dt.date
    reduceMemory('customer_id', train)

    # delete price and sales_channel_id value
    train = train.drop('price', 1)
    train = train.drop('sales_channel_id', 1)

    # find customer who purchase more than 1 times
    a = pd.DataFrame(train.loc[train['t_dat'] == date(2020, 9, 21), 'customer_id'])
    a = a.drop_duplicates()
    b = train.loc[train['t_dat'] == date(2020, 9, 21)]
    b = pd.DataFrame(b.loc[b.customer_id.isin(a.customer_id), 'customer_id'])
    b = b.drop_duplicates()
    # split transaction into years and generate csv
    tx2018 = train.loc[train['t_dat'] <= date(2019, 9, 20)]
    tx2019 = train.loc[(date(2019, 9, 20) < train['t_dat']) & (train['t_dat'] <= date(2020, 9, 20))]
    tx2020 = train.loc[(date(2020, 9, 20) < train['t_dat']) & (train['t_dat'] <= date(2021, 9, 20))]
    tx2018.to_csv('transaction_2018.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2019.to_csv('transaction_2019.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2020.to_csv('transaction_2020.csv', sep=',', encoding='UTF-8', index=None, header=True)


def article(train: []):
    # delete nonnumerical  value
    train = train.drop('prod_name', 1)
    train = train.drop('product_type_name', 1)
    train = train.drop('product_group_name', 1)
    train = train.drop('graphical_appearance_name', 1)
    train = train.drop('colour_group_name', 1)
    train = train.drop('perceived_colour_value_name', 1)
    train = train.drop('perceived_colour_master_name', 1)
    train = train.drop('department_name', 1)
    train = train.drop('index_code', 1)
    train = train.drop('index_name', 1)
    train = train.drop('index_group_name', 1)
    train = train.drop('section_name', 1)
    train = train.drop('garment_group_name', 1)
    train = train.drop('detail_desc', 1)

    # check and delete missing value
    train = train.dropna()

    train.to_csv('training_article.csv', sep=',', encoding='UTF-8', index=None, header=True)


data1 = pd.read_csv('/Users/nakreond/Programming_Languages/Python_Projects/H&M Personalized Fashion Recommendations/customers.csv')
data2 = pd.read_csv('/Users/nakreond/Programming_Languages/Python_Projects/H&M Personalized Fashion Recommendations/H_M_transaction_2020.csv')
data3 = pd.read_csv('/Users/nakreond/Programming_Languages/Python_Projects/H&M Personalized Fashion Recommendations/articles.csv')

customer(data1)
transaction(data2)
article(data3)
