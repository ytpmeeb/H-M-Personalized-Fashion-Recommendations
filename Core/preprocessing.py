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
    # pd.unique(test['club_member_status'])

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

    data.to_csv('dataing_customer.csv', sep=',', encoding='UTF-8', index=None, header=True)


def transaction_data(data: []):
    data['t_dat'] = pd.to_datetime(data['t_dat'])
    data['t_dat'] = data['t_dat'].dt.date
    reduceMemory('customer_id', data)

    # delete price and sales_channel_id value
    data = data.drop('price', 1)
    data = data.drop('sales_channel_id', 1)

    # find customer who purchase more than 1 times
    a = pd.DataFrame(data.loc[data['t_dat'] == date(2020, 9, 21), 'customer_id'])
    a = a.drop_duplicates()
    b = data.loc[data['t_dat'] == date(2020, 9, 21)]
    b = pd.DataFrame(b.loc[b.customer_id.isin(a.customer_id), 'customer_id'])
    b = b.drop_duplicates()
    # split transaction into years and generate csv
    tx2018 = data.loc[data['t_dat'] <= date(2019, 9, 20)]
    tx2019 = data.loc[(date(2019, 9, 20) < data['t_dat']) & (data['t_dat'] <= date(2020, 9, 20))]
    tx2020 = data.loc[(date(2020, 9, 20) < data['t_dat']) & (data['t_dat'] <= date(2021, 9, 20))]
    tx2018.to_csv('transaction_2018.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2019.to_csv('transaction_2019.csv', sep=',', encoding='UTF-8', index=None, header=True)
    tx2020.to_csv('transaction_2020.csv', sep=',', encoding='UTF-8', index=None, header=True)


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

    data.to_csv('training_article.csv', sep=',', encoding='UTF-8', index=None, header=True)


# data1 = pd.read_csv('.../customers.csv')
data1 = pd.read_csv('.../customers.csv')

# data2 = pd.read_csv('.../H_M_transaction_2020.csv')
# data2 = pd.read_csv('.../transactions_train.csv')

# data3 = pd.read_csv('.../articles.csv')
data3 = pd.read_csv('.../articles.csv')

customer_data(data1)
# transaction_data(data2)
article_data(data3)
