from Tools import Anomaly_Detection
from Tools import Data_Integration
from Tools import Data_Reduction
from Tools import Data_Transformation
from Tools import Supervised_Learning
from Tools import Unsupervised_Learning
from sklearn.metrics import multilabel_confusion_matrix
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


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Bootstrap Aggregation Algorithm
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)



"""
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
    """



"""
from Tools import Unsupervised_Learning
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
tran20L = pd.read_csv('./Data/Training Data/year/training_transaction_2020_last_week.csv')
customerM = pd.read_csv('./Data/Training Data/training_customer_with_multi-purchases.csv')
article = pd.read_csv('./Data/Training Data/training_article.csv')

test = group_transaction(tran20L, article, customerM)
p, r = Unsupervised_Learning.fp_growth(test['transaction'], 5, 0.9)

test[[14 in x for x in test[0]]]
"""






import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.array([[0,0,1], [1,1,0],[0,1,0]])
y_pred = np.array([[0,0,1], [1,0,1],[1,0,0]])

labels = ["A", "B", "C"]

conf_mat_dict={}

for label_col in range(len(labels)):
    y_true_label = y_true[:, label_col]
    y_pred_label = y_pred[:, label_col]
    conf_mat_dict[labels[label_col]] = confusion_matrix(y_pred=y_pred_label, y_true=y_true_label)


for label, matrix in conf_mat_dict.items():
    print("Confusion matrix for label {}:".format(label))
    print(matrix)















# Bagging Algorithm on the Sonar dataset
from random import seed
from random import randrange
from csv import reader


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            # for i in range(len(dataset)):
            # 	row = dataset[randrange(len(dataset))]
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# Bootstrap Aggregation Algorithm
def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)

# Test bagging on the sonar dataset
seed(1)
# load and prepare data
dataset = load_csv('C:/Users/naerkond/Downloads/sonar.all-data.csv')
# convert string attributes to integers
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
max_depth = 6
min_size = 2
sample_size = 0.50
for n_trees in [1, 5, 10, 50]:
	scores = evaluate_algorithm(dataset, bagging, n_folds, max_depth, min_size, sample_size, n_trees)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

