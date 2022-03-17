import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd


article_csv = '/Users/nakreond/Programming_Languages/Python_Projects/' \
           'H&M Personalized Fashion Recommendations/articles.csv'
customer_csv = '/Users/nakreond/Programming_Languages/Python_Projects/' \
           'H&M Personalized Fashion Recommendations/customers.csv'
tx_csv = '/Users/nakreond/Programming_Languages/Python_Projects/' \
           'H&M Personalized Fashion Recommendations/transactions_train.csv'
wage_csv = '/Users/nakreond/Programming_Languages/Python_Projects/' \
           'H&M Personalized Fashion Recommendations/Iowa_Wage_Data_by_Occupation.csv'

article_df = pd.read_csv(article_csv)
article_df.head()
print("shape of data article data", article_df.shape)

customer_df = pd.read_csv(customer_csv)
customer_df.head()
print("shape of data customer data", customer_df.shape)

tx_df = pd.read_csv(tx_csv)
tx_df.head()
print("shape of data transactions data", tx_df.shape)


# Function to plot the Nan percentages of each columns
def plot_nas(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100

        # rearrange the last col
        # cols = df.columns.to_list()
        # cols = cols[-1:] + cols[:-1]
        # na_df = na_df[cols]

        # delete the 0 % col
        # na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)

        missing_data = pd.DataFrame({'Missing Ratio %': na_df})
        missing_data.plot(kind="barh")
        plt.show()
    else:
        print('No NAs found')


print("Checking Null's in data")
plot_nas(article_df)
plot_nas(customer_df)
plot_nas(tx_df)


def plot_bar(df, column):
    long_df = pd.DataFrame(
        df.groupby(column)['customer_id'].count().reset_index().rename({'customer_id': 'count'},
                                                                       axis=1))
    fig = px.bar(long_df, x=column, y="count", color=column, title=f"bar plot for {column} ")
    fig.show()


def plot_hist(df, column):
    fig = px.histogram(df, x=column, nbins=10, title=f'{column} distribution ')
    fig.show()


def plot_bar(df, column):
    long_df = pd.DataFrame(
        df.groupby(column)['article_id'].count().reset_index().rename({'article_id': 'count'},
                                                                      axis=1))
    fig = px.bar(long_df, x=column, y="count", color=column, title=f"bar plot for {column} ")
    fig.show()


def plot_hist(df, column):
    fig = px.histogram(df, x=column, nbins=10, title=f'{column} distribution ')
    fig.show()


plot_bar(customer_df, 'age')
plot_bar(customer_df, 'postal_code')
plot_bar(customer_df, 'product_type_name')
plot_bar(customer_df, 'product_group_name')
plot_bar(customer_df, 'graphical_appearance_name')
plot_bar(customer_df, 'index_name')
plot_bar(customer_df, 'garment_group_name')
