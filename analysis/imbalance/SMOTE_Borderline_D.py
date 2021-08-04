from typing import Counter
from numpy.lib.type_check import nan_to_num
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.neighbors import NearestNeighbors
from base_sampler import *
import numpy as np
import pandas as pd
import math
from sklearn.utils import _safe_indexing


ROOT_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/analysis/imbalance/'

# 处于多数类与少数类边缘的样本
def in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data):
    nn_m = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data)
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nnm_x = NearestNeighbors(n_neighbors=11).fit(imbalanced_featured_data).kneighbors(old_feature_data,
                                                                                    return_distance=False)[:,1:]
    nn_label = (imbalanced_label_data[nnm_x] != old_label_data).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return np.bitwise_and(n_maj >= (nn_m.n_neighbors - 1) / 2, n_maj < nn_m.n_neighbors - 1)


# 使用K-近邻方法产生新样本
def make_sample(old_feature_data, danger_index_data,n):
    print('需生成少数类样本个数：',n)
    b = 1
    n = int(b * n)
    k = 5 # k 近邻
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nns2 = NearestNeighbors(n_neighbors=k+1).fit(old_feature_data).kneighbors(danger_index_data, return_distance=True)
    nns = nns2[1][:,1:]
    # 2
    d = nns2[0][:,1:] # object[i] 与每个最近邻的距离
    # object[i] 与最近邻的距离的标准差
    
    r = np.zeros((d.shape[0], 1))
    for key,value in enumerate(d):
        # 3
        r[key][0] = (np.std(value, ddof = 1))
    # 4
    r_sum = np.sum(r)
    pi = np.zeros((d.shape[0], 1))
    for key,value in enumerate(r):
        pi[key][0] = value / r_sum
    # 5
    pij = np.zeros((d.shape[0], d.shape[1]))
    for key,value in enumerate(d):
        pij[key] = value / np.sum(value)
    # 6
    sij = pi * pij * n
    s = 0
    for key, value in enumerate(sij):
        l = sij[key]
        for i in range(len(l)):
            l[i] = round(l[i])
            s += l[i]
    print('实际生成样本数：',s)
    # 7 TODO
    new_feature_data = []
    for key, value in enumerate(danger_index_data):
        l_obji = [] 
        nns_index = nns[key]
        for i in range(k):
            l_obji.append(danger_index_data[key])
            
        l_objij = old_feature_data[nns_index]
        obji = np.array(l_obji)
        objij = np.array(l_objij)
        diffij = objij - obji
        # 8
        sij_value = sij[key]
        for i,j in enumerate(diffij):
            diffij_d_value = diffij[i] / (sij_value[i] + 1)
            # 9
            sij_obji = sij[key]
            sij_obji_k = sij_obji[i]
            if sij_obji_k <= 0:
                continue
            for l in range(int(sij_obji_k)): 
                new_sij = danger_index_data[key] + diffij_d_value * (l+1)
                new_feature_data.append(new_sij)
    # 将原少数类样本点与新产生的少数类样本点整合
    new_feature_data = np.array(new_feature_data)
    new_min_feature_data = np.vstack((new_feature_data, old_feature_data))
    # TODO 数量少了一点
    return new_min_feature_data
        

def SMOTE_Borderline_D(imbalanced_data_arr2):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    print('多数类样本数：',len(major_data_arr2),', 少数类样本数：',len(minor_data_arr2))
    imbalanced_featured_data = imbalanced_data_arr2[:, : -1]
    imbalanced_label_data = imbalanced_data_arr2[:, -1]
    # 计算多数类数据和少数类数据之间的数量差,也是需要过采样的数量
    # 1
    n = major_data_arr2.shape[0] - minor_data_arr2.shape[0]

    # 原始少数样本的特征集
    old_feature_data = minor_data_arr2[:, : -1]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][-1]

    danger_index = in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data)
    # 少数样本中噪音集合，也就是最终要产生新样本的集合
    danger_index_data = _safe_indexing(old_feature_data, danger_index)

    # 使用K近邻方法产生的新样本特征集
    new_feature_data = make_sample(old_feature_data, danger_index_data, n) # 扩展少数类样本集
    # 将类别标签数组合并到少数类样本特征集，构建出新的少数类样本数据集
    new_labels_data = np.array([old_label_data] * len(new_feature_data))
    new_minor_data_arr2 = np.column_stack((new_feature_data, new_labels_data))
    
    # balanced_data_arr2 = np.row_stack((new_minor_data_arr2, major_data_arr2))
    balanced_data_arr2 = np.row_stack((major_data_arr2, new_minor_data_arr2))
    # 将少数类数据集和多数据类数据集合并，并对样本数据进行打乱重排，
    # balanced_data_arr2 = concat_and_shuffle_data(new_minor_data_arr2, major_data_arr2)
    return balanced_data_arr2

def test():
    imbalanced_data = np.load(ROOT_PATH + 'imbalanced_train_data_arr2.npy')
    # print(imbalanced_data.shape)
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data)

    # print(minor_data_arr2.shape)
    # print(major_data_arr2.shape)
    # # 测试SMOTE方法
    balanced_data_arr2 = SMOTE_Borderline_D(imbalanced_data)
    print(balanced_data_arr2)
    print(balanced_data_arr2.shape)

if __name__ == '__main__':
    test()

