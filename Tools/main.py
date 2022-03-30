from Tools import Data_Integration
from Tools import Data_Reduction
from Tools import Data_Transformation
from Tools import Anomaly_Detection
from Tools import Supervised_Learning
from Tools import Unsupervised_Learning


def integration():
    # Enter the value
    x = [1, 1, 1, 2, 2, 1, 0, 0, 0, 0]
    y = [0, 1, 0, 2, 2, 0, 1, 0, 0, 0]

    euclidean_distance = Data_Integration.Similarity(x, y).Euclidean_distance()
    manhattan_distance = Data_Integration.Similarity(x, y).Manhattan_distance()
    minkowski_distance = Data_Integration.Similarity(x, y).Minkowski_distance()
    cosine_similarity = Data_Integration.Similarity(x, y).Cosine_similarity()
    cosine_distance = Data_Integration.Similarity(x, y).Cosine_distance()
    pearson_correlation_coefficient = Data_Integration.Correlation(x, y).Pearson_correlation_coefficient()

    print('------------Similarity------------')
    print('Euclidean Distance between a and b is: ', euclidean_distance[0])
    print('Manhattan Distance between a and b is: ', manhattan_distance[0])
    print('Minkowski Distance between a and b is: ', minkowski_distance[0])
    print('Cosine Similarity between a and b is: ', cosine_similarity[0])
    print('Cosine Distance between a and b is: ', cosine_distance[0])
    print('------------Correlation------------')
    print('Pearson Correlation Coefficient between a and b is: ', pearson_correlation_coefficient[0][0])


integration()

