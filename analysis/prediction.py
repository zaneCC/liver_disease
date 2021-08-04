'''
    肝病预测模型
    结果：
    使用10折交叉验证
    使用lasso特征选择保留特征：[双下肢水肿,乏力,黑便,腹痛,身目黄染,咳痰,全身胀痛,口苦,行走困难,欲呕,肝区疼痛]
    使用lasso特征选择后：           拆分身目尿黄=身目黄染+尿黄  未做特征选择
        逻辑回归：      81.24%   81.99%                     81.99 %
        支持向量机：    81.61%   81.62%                     80.46 %
        随机森林：      80.84%   81.23%                     81.23 %
        决策树：        80.46%   80.07%                     76.61 %
        adaboost：     82.76%   82.76%                      80.48 %
    使用逻辑回归+l1正则特征选择：[腹胀,尿黄,纳差,双下肢水肿,乏力,黑便,头晕,腹痛,身目黄染,恶心,腹泻,呕吐,精神差,口干,水样便]
        逻辑回归：      81.99%
        支持向量机：    80.07%
        随机森林：      81.99%
        决策树：        78.53%
        adaboost：     81.24%
    
'''
OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
from typing import Counter
# import pydotplus,graphviz
# from IPython.display import Image
from sklearn import tree
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants
import utils.misc as misc
from sklearn.metrics import accuracy_score,confusion_matrix 
from sklearn.model_selection import train_test_split,validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib as mlp
import re 
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB


# 汇总表-未做特征选择
# PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
# PATH = constants.SELECTION_MERGE_CSV_PATH
# 人工选择特征
# PATH = constants.SYMP_MAIN_ACC_DIAGNOSIS_PATH
# 汇总表-SMOTE过采样后
# PATH = constants.SMOTE_MERGE_CSV_PATH
# SMOTE Borderline1 采样
# PATH = constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH
# SMOTE_D 采样
# PATH = constants.SMOTE_D_MERGE_CSV_PATH
# SMOTE_BORDERLINE_D 采样
PATH = constants.SMOTE_Borderline_D_CSV_PATH

# 可视化
TO_VIEW_PATH = OS_WINDOWS_PATH + '/output/决策树可视化.pdf'
OLD_PATH = OS_WINDOWS_PATH + '/output/dot/dot_data.txt'
NEW_PATH = OS_WINDOWS_PATH + '/output/dot/dot_data_new.txt'

class Prediction():
    
    def __init__(self):
        self.misc = misc.Misc()
        self.isShow = True # 是否展示图像

        df = pd.read_csv(PATH)
        cols = df.columns.values.tolist()
        # cols.remove('Unnamed: 0')
        # cols.remove('INHOSPTIAL_ID')
        cols.remove('ZHENGHOU1')

        self.X = df[cols] 
        self.y = df['ZHENGHOU1']

        self.split()

    
    def split(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size = 0.2,random_state=0)
        ss = StandardScaler()
        self.X_train = ss.fit_transform(X_train)
        self.X_test = ss.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def NB(self):
        return GaussianNB()

    # 逻辑回归
    def logisiticRegression(self):   
        lg=LogisticRegressionCV(multi_class="multinomial", solver="newton-cg")
        return lg

    # 支持向量机
    def SVM(self):
        sv=SVC(kernel='linear',random_state=0, probability=True)
        return sv

    # knn
    def knn(self):
        score=[]

        for i in range(1,10):
            knn=KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)
            knn.fit(self.X_train,self.y_train)
            pre3=knn.predict(self.X_test)
            ans=accuracy_score(pre3,self.y_test)
            score.append(round(100*ans,2))
        print('------------- 训练数据的评价---- knn')
        print(sorted(score,reverse=True)[:5])
        knn=sorted(score,reverse=True)[:1]
        return knn

    # 决策树
    def decisionTree(self):
        # dt = DecisionTreeClassifier(max_depth=5)
        dt = DecisionTreeClassifier()
        return dt

    # 随机森林
    def randomForestClassifier(self):
        # 未调参  80.84 %
        # rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')

        # 使用调参后的模型 81.61 %
        # {'criterion': 'gini','max_depth': 3, 'max_features': 0.4,'min_samples_split': 3,'n_estimators': 10}
        return RandomForestClassifier(n_estimators=10, criterion='gini', max_features=0.4, min_samples_split=3, max_depth=3)

    # adaboost
    def adaboostClassifier(self):
        # 未调参 82.38 %
        # return AdaBoostClassifier(n_estimators=100)

        # 调参后  82.76 %
        # 'algorithm': 'SAMME.R'  'learning_rate': 0.2  'n_estimators': 2, SAMME.R > SAMME
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.2, algorithm='SAMME.R')

    def fit(self,regs):
        for key,(name, i) in enumerate(regs):
            i.fit(self.X_train,self.y_train)
        
    def predict(self,regs):
        for key,(name, i) in enumerate(regs):
            # score = i.score(self.X_test,self.y_test)
            # print('score:',score)
            pre = i.predict(self.X_test)
            # ans = accuracy_score(pre,self.y_test)
            # print(name,ans * 100)
            print('模型:',name)
            print(classification_report(pre,self.y_test))

    # 交叉验证
    # 问题： 因为数据不平衡，使用交叉验证有可能拆分结果仍不均衡，需要输出每轮交叉验证中不同类别拆分的样本数
    # 解答： 不用担心，官方文档写了“如果估计器是分类器并且y是二元或多类，StratifiedKFold则使用”
    def crossScore(self,regs):
        color = ['r','g','b','c','y']
        for key,(name, i) in enumerate(regs):
            print('------------- 【评价】测试数据 ---------------',i)
            accuracies = cross_val_score(estimator=i, X=self.X ,y=self.y,cv=10)
            self.misc.plotAccuLine(accuracies*100,color=color[key],dlabel=name)

            print('accuracies:',accuracies)
            print("accuracy is {:.2f} %".format(accuracies.mean()*100))
            print("std is {:.2f} %".format(accuracies.std()*100)) # 标准偏差估计分数
        if self.isShow:
            self.misc.show()

    def pipline(self,estimators):
        # pipe_lr = LogisticRegression(random_state=1)
        for i in estimators:
            pipe_lr = Pipeline([('sc', StandardScaler()),
                        ('pca', PCA(n_components=10)),
                        ('clf', i)
                        ])
            pipe_lr.fit(self.X_train, self.y_train)
            accuracies = pipe_lr.score(self.X_test, self.y_test)
            print("accuracy is {:.2f} %".format(accuracies.mean()*100))
    
    # 决策树可视化
    # def viewTree(self):
        # clf = self.decisionTree()
        # #拟合模型
        # clf.fit(self.X_train, self.y_train)

        # dot_data = tree.export_graphviz(clf, out_file=None,
        #                         feature_names=self.X.columns,
        #                         class_names='ZHENGHOU1',
        #                         filled=True, rounded=True,
        #                         special_characters=True)
        # graph = pydotplus.graph_from_dot_data(dot_data)
        # # graph [fontname="Microsoft Yahei"]
        # # 使用ipython的终端jupyter notebook显示。
        # # Image(graph.create_png())
        # # 如果没有ipython的jupyter notebook，可以把此图写到pdf文件里，在pdf文件里查看。
        # # graph.write_pdf(TO_VIEW_PATH) 

        # f = open(OLD_PATH, 'w') 
        # f.write(dot_data) 
        # f.close()

    # 解决决策树可视化乱码问题
    def font_conf(self, f_old, f_new, filename, view_path):
        for line in f_old: 
            if 'fontname' in line:         
                font_re = 'fontname=(.*?)]'         
                old_font = re.findall(font_re, line)[0]
                line = line.replace(old_font, 'SimHei')     
            f_new.write(line) 
        
        for line in f_old: 
            if 'fontname' in line: 
                font_re = 'fontname=(.*?)]' 
                old_font = re.findall(font_re, line)[0] 
                line = line.replace(old_font, 'SimHei') 
            f_new.write(line)
        f_old.close()
        f_new.close()
        
        # os.system('dot -Tpdf dot_data_new.txt -o ' + TO_VIEW_PATH)
        os.system('dot -Tpdf ' + filename + ' -o ' + view_path)

    # 随机森林可视化（输出所有决策树PDF图）
    # def test_RF(self):
    #     clf = pre.randomForestClassifier()
    #     clf.fit(self.X_train, self.y_train)

    #     # 将所有决策树输出
    #     Estimators = clf.estimators_
    #     treeCount = 0
    #     index_paths = []
    #     root_path = OS_WINDOWS_PATH + '/output/dot/dot_data_'
    #     # 写入绘制文件
    #     for index, model in enumerate(Estimators):
    #         # filename = 'iris_' + str(index) + '.pdf'
    #         dot_data = tree.export_graphviz(model , out_file=None,
    #                             feature_names=self.X.columns,
    #                             class_names='ZHENGHOU1',
    #                             filled=True, rounded=True,
    #                             special_characters=True)
    #         graph = pydotplus.graph_from_dot_data(dot_data)
    #         # 使用ipython的终端jupyter notebook显示。
    #         path = root_path + str(index) + '.txt'
    #         index_paths.append(index)
    #         f = open(path, 'w') 
    #         f.write(dot_data) 
    #         f.close()
    #         treeCount += 1

        # 保存PDF图片
        # for i in range(treeCount):
        #     f_old = open(root_path + str(index_paths[i]) + '.txt','r')
        #     f_new = open(root_path + 'new_' + str(index_paths[i]) + '.txt', 'w', encoding='utf-8') 
        #     filename = OS_WINDOWS_PATH + '/output/dot/' + 'dot_data_new_' + str(index_paths[i]) + '.txt'
        #     to_path =  OS_WINDOWS_PATH + '/output/rf/tree/' +  'dot_data_new_' + str(index_paths[i]) + '.pdf'

        #     self.font_conf(f_old,f_new, filename, to_path)

    # 使用随机森林，输出特征重要性
    def importantFeaturesRF(self):
        clf = pre.randomForestClassifier()
        clf.fit(self.X_train, self.y_train)
        feat_labels = self.X.columns
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(self.X_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

    def roc(self,regs):
        self.misc.figure()
        color = ['r','g','b','c','y']
        for key,(name, i) in enumerate(regs):
            if name == 'Random Forest' or name == 'Decision Tree' or 'NB':  # 这两个分类器没有 decision_function 方法
                y_pred_proba = i.fit(self.X_train, self.y_train).predict_proba(self.X_test)
                fpr,tpr,threshold = roc_curve(self.y_test, y_pred_proba[:,1], pos_label=2) # 计算真正率和假正率
            else:
                y_pred_proba = i.fit(self.X_train, self.y_train).decision_function(self.X_test)
                fpr,tpr,threshold = roc_curve(self.y_test, y_pred_proba, pos_label=2) # 计算真正率和假正率
            
            roc_auc = auc(fpr,tpr)
            title = '原始数据\'s ROC'
            if PATH == constants.SMOTE_MERGE_CSV_PATH:
                title = 'SMOTE \'s ROC'
            elif PATH == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH:
                title = 'SMOTE Borderline\'s ROC'
            elif PATH == constants.SMOTE_D_MERGE_CSV_PATH:
                title = 'SMOTE D\'s ROC'
            elif PATH == constants.SMOTE_Borderline_D_CSV_PATH:
                title = 'SMOTE Borderline D\'s ROC'
            self.misc.plotAUC(fpr, tpr, roc_auc, color=color[key], label=name, title = title)

        self.misc.show()


    
if __name__ == "__main__":
    pre = Prediction()

    # 预测
    regs = [
            ('Logisitic RegressionCV',pre.logisiticRegression()),
            ('SVC',pre.SVM()),
            ('Random Forest',pre.randomForestClassifier()),
            ('Decision Tree',pre.decisionTree()),
            ('Adaboost',pre.adaboostClassifier()),
            # ('NB', pre.NB())
            ]
    # estimators = [pre.logisiticRegression(), pre.SVM(), pre.decisionTree(), pre.randomForestClassifier(), pre.adaboostClassifier()]
    pre.fit(regs)
    pre.predict(regs)

    # 交叉验证
    # pre.crossScore(regs)
    # PCA和标准化
    # pre.pipline(estimators)

    pre.roc(regs)
    
    # 决策树可视化
    # pre.viewTree()
    # f_old = open(OLD_PATH, 'r') 
    # f_new = open(NEW_PATH, 'w', encoding='utf-8') 
    # filename = 'dot_data_new.txt'
    # pre.font_conf(f_old,f_new, filename, TO_VIEW_PATH)

    # 随机森林可视化
    # pre.test_RF()

    # 输出特征重要性
    # pre.importantFeaturesRF()


