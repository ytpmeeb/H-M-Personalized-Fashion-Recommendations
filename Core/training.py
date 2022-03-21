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


class Clustering:
    @staticmethod
    def customer(data):
        # skip the customer ID
        m_nor = Data_Transformation.min_max(data.iloc[:, 1:5])
        z_nor = Data_Transformation.zScore(data.iloc[:, 1:5])
        m_nor_data = Data_Reduction.pca(m_nor)
        z_nor_data = Data_Reduction.pca(z_nor)
        # Unsupervised_Learning.k_mean_find_k(m_nor_data)
        k = Unsupervised_Learning.k_mean(m_nor_data, 10)

    @staticmethod
    def article(data):
        # skip the article ID
        m_nor = Data_Transformation.min_max(data.iloc[:, 1:11])
        z_nor = Data_Transformation.zScore(data.iloc[:, 1:11])
        m_nor_data = Data_Reduction.pca(m_nor)
        z_nor_data = Data_Reduction.pca(z_nor)
        # Unsupervised_Learning.k_mean_find_k(m_nor_data)
        k = Unsupervised_Learning.k_mean(m_nor_data, 10)

# customer train
data = pd.read_csv('.../HnM_Personalized_Fashion_Recommendations/Datasets/Training/training_customer.csv')
# article train
data = pd.read_csv('.../HnM_Personalized_Fashion_Recommendations/Datasets/Training/training_article.csv')
