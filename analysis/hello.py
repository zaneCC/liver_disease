from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
import constants
from sklearn.cluster import KMeans,SpectralClustering,MiniBatchKMeans
from sklearn import datasets
from sklearn import metrics


# 汇总表-SMOTE过采样后
# PATH = constants.SMOTE_MERGE_CSV_PATH
# SMOTE Borderline1 采样
# PATH = constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH
# SMOTE_D 采样
# PATH = constants.SMOTE_D_MERGE_CSV_PATH
# SMOTE_BORDERLINE_D 采样
PATH = constants.SMOTE_Borderline_D_CSV_PATH

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

df = pd.read_csv(PATH)
cols = df.columns.values.tolist()
cols.remove('ZHENGHOU1')
X = df[cols] 
y = df['ZHENGHOU1']
X,y = shuffle_data(X.to_numpy(),y.to_numpy())
# print(X)
# print(y)

# cls = KMeans(n_clusters=2, init='k-means++')
# y_hat = cls.fit_predict(X)
# print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, y_hat))

# print(y_hat)
# print(y)
# count_true = 0
# for i,j in enumerate(y_hat):
#     if y_hat[i] == 0 and y[i] == 1:
#         count_true += 1
#         continue
#     if y_hat[i] == 1 and y[i] == 2:
#         count_true += 1
#         continue

# print('y: ',len(y), 'count_true: ',count_true)

# for index, k in enumerate((2,3,4,5,6,7,8)):
#     # plt.subplot(2,2,index+1)
#     y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(X)
#     score= metrics.calinski_harabasz_score(X, y_pred)  
#     print('k:',k,',score:',score)

# for index, gamma in enumerate((0.01,0.1,1,10)):
#     for index, k in enumerate((3,4,5,6)):
#         y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
#         print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k,"score:", metrics.calinski_harabasz_score(X, y_pred))


# X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
# from sklearn.cluster import SpectralClustering
# y_pred = SpectralClustering().fit_predict(X)
# from sklearn import metrics
# print("Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, y_pred))
