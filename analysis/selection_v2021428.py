OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'

import pandas as pd
import numpy as np
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/')
import constants
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold # 通过方差提取特征
from sklearn.decomposition import PCA

PATH = constants.MERGE_CSV_PATH
TO_PATH = constants.SELECTION_MERGE_CSV_PATH


class Selection_v2021428():

    def __init__(self):
        df = pd.read_csv(PATH)
        cols = df.columns.values.tolist()
        # cols.remove('Unnamed: 0')
        cols.remove('INHOSPTIAL_ID')
        cols.remove('ZHENGHOU1')
        self.X = df[cols]
        self.y = df['ZHENGHOU1']
        self.df = df


    def split(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        return X_train,X_test,y_train,y_test

    # lasso 特征选择
    def lasso(self):
        X_train,X_test,y_train,y_test = self.split(self.X, self.y)

        model = Lasso(alpha=0.05, fit_intercept=False)
        model.fit(X_train, y_train)

        print(model.coef_.ravel())

        return model

    # 选择特征
    def selection(self, model):
        count = 0
        for index, row in self.df.iteritems():
            if index == 'INHOSPTIAL_ID' or index == 'ZHENGHOU1':
                continue
            if model.coef_.ravel()[count] > 0.00001 or model.coef_.ravel()[count] < -0.00001:
                count += 1
                print(index)
                continue
            self.df.drop(index,axis=1,inplace=True)
            count += 1

    def KBest(self):
        print('-------------各个症状与诊断结果进行卡方检验')
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        best_feature = SelectKBest(score_func=chi2,k=10)
        # 各个症状与诊断结果进行卡方检验
        fit = best_feature.fit(self.X,self.y)
        # 得出卡方检验分数
        dataset_scores = pd.DataFrame(fit.scores_)
        dataset_cols = pd.DataFrame(self.X.columns)

        featurescores = pd.concat([dataset_cols,dataset_scores],axis=1)
        featurescores.columns=['column','scores']
        # 输出贡献最大的十个症状
        print(featurescores.nlargest(10,'scores'))
        # 画出图形显示
        # featureview=pd.Series(fit.scores_, index=self.X.columns)
        # featureview.plot(kind='barh')
        return featurescores.nlargest(10,'scores')

    def KBestDF(self, fList):
        return self.df[fList]

    def PCA(self):
        pca=PCA(n_components=10)
        newX=pca.fit_transform(self.X)
        X = pca.inverse_transform(newX)
        print(X)

    def featureSelectionByVariance(self):
        print('-------------通过方差提取特征')

        feature_high_variance = VarianceThreshold(threshold=(0.5*(1-0.5)))
        falls=feature_high_variance.fit(self.X)

        dataset_scores1 = pd.DataFrame(falls.variances_)
        dat1 = pd.DataFrame(self.X.columns)

        high_variance = pd.concat([dataset_scores1,dat1],axis=1)
        high_variance.columns=['variance','cols']
        print(high_variance[high_variance['variance']>0.2])


    # 输出 CSV 文件
    def toCSV(self,path,df=None):
        if df is None:
            self.df.to_csv(path)
        else:
            df.to_csv(path)

if __name__  == '__main__':
    sele = Selection_v2021428()
    # 特征选择：Lasso
    model = sele.lasso()
    sele.selection(model)
    sele.toCSV(TO_PATH)

    # 特征选择：卡方检验
    # featurescores = sele.KBest()
    # fList = list(featurescores['column'].values)
    # fList.append('ZHENGHOU1')
    # retDF = sele.KBestDF(fList)
    # sele.toCSV(TO_PATH,retDF)

    # 特征选择：PCA
    # sele.PCA()
