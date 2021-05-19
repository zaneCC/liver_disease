# 肝病预测模型
OS_MAC_PATH = '/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'
import pydotplus,graphviz
from IPython.display import Image
from sklearn import tree
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r'E:/liver_disease/liver_disease')
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

# 汇总表-未做特征选择
# PATH = constants.MERGE_CSV_PATH
# 汇总表-特征选择后
PATH = constants.SELECTION_MERGE_CSV_PATH
# 人工选择特征
# PATH = constants.SYMP_MAIN_ACC_DIAGNOSIS_PATH

# 可视化
TO_VIEW_PATH = OS_WINDOWS_PATH + '/output/决策树可视化.pdf'
OLD_PATH = OS_WINDOWS_PATH + '/output/dot/dot_data.txt'
NEW_PATH = OS_WINDOWS_PATH + '/output/dot/dot_data_new.txt'

class Prediction():
    
    def __init__(self):
        self.misc = misc.Misc()
        self.isShow = False

        df = pd.read_csv(PATH)
        cols = df.columns.values.tolist()
        # cols.remove('Unnamed: 0')
        cols.remove('INHOSPTIAL_ID')
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

    # 逻辑回归
    def logisiticRegression(self):   
        lg=LogisticRegressionCV(multi_class="multinomial", solver="newton-cg")
        return lg

    # 支持向量机
    def SVM(self):
        sv=SVC(kernel='linear',random_state=0)
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
        dt = DecisionTreeClassifier(max_depth=5)
        return dt

    # 随机森林
    def randomForestClassifier(self):
        rfc = RandomForestClassifier(n_estimators=100, criterion='entropy')
        # rfc.fit(self.X_train,self.y_train)
        return rfc

    # adaboost
    def adaboostClassifier(self):
        return AdaBoostClassifier(n_estimators=100)

    def fit(self,regs):
        for key,(name, i) in enumerate(regs):
            i.fit(self.X_train,self.y_train)
        
    def predict(self,regs):
        for key,(name, i) in enumerate(regs):
            # score = i.score(self.X_test,self.y_test)
            # print('score:',score)
            pre = i.predict(self.X_test)
            ans = accuracy_score(pre,self.y_test)
            print(name,ans * 100)

    # 交叉验证
    def crossScore(self,regs):
        color = ['r','g','b','c','y']
        for key,(name, i) in enumerate(regs):
            print('------------- 【评价】测试数据 ---------------',i)
            accuracies = cross_val_score(estimator=i, X=self.X ,y=self.y,cv=10)
            self.misc.plotAccuLine(accuracies,color=color[key],dlabel=name)

            # print('accuracies:',accuracies)
            print("accuracy is {:.2f} %".format(accuracies.mean()*100))
            # print("std is {:.2f} %".format(accuracies.std()*100)) # 标准偏差估计分数
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
    def viewTree(self):
        clf = self.decisionTree()
        #拟合模型
        clf.fit(self.X_train, self.y_train)

        dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=self.X.columns,
                                class_names='ZHENGHOU1',
                                filled=True, rounded=True,
                                special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # graph [fontname="Microsoft Yahei"]
        # 使用ipython的终端jupyter notebook显示。
        # Image(graph.create_png())
        # 如果没有ipython的jupyter notebook，可以把此图写到pdf文件里，在pdf文件里查看。
        # graph.write_pdf(TO_VIEW_PATH) 

        f = open(OLD_PATH, 'w') 
        f.write(dot_data) 
        f.close()

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

    def test_RF(self):
        clf = pre.randomForestClassifier()
        clf.fit(self.X_train, self.y_train)

        # 将所有决策树输出
        Estimators = clf.estimators_
        treeCount = 0
        index_paths = []
        root_path = OS_WINDOWS_PATH + '/output/dot/dot_data_'
        # 写入绘制文件
        for index, model in enumerate(Estimators):
            # filename = 'iris_' + str(index) + '.pdf'
            dot_data = tree.export_graphviz(model , out_file=None,
                                feature_names=self.X.columns,
                                class_names='ZHENGHOU1',
                                filled=True, rounded=True,
                                special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            # 使用ipython的终端jupyter notebook显示。
            path = root_path + str(index) + '.txt'
            index_paths.append(index)
            f = open(path, 'w') 
            f.write(dot_data) 
            f.close()
            treeCount += 1

        # 保存PDF图片
        for i in range(treeCount):
            f_old = open(root_path + str(index_paths[i]) + '.txt','r')
            f_new = open(root_path + 'new_' + str(index_paths[i]) + '.txt', 'w', encoding='utf-8') 
            filename = OS_WINDOWS_PATH + '/output/dot/' + 'dot_data_new_' + str(index_paths[i]) + '.txt'
            to_path =  OS_WINDOWS_PATH + '/output/rf/tree/' +  'dot_data_new_' + str(index_paths[i]) + '.pdf'

            self.font_conf(f_old,f_new, filename, to_path)

    # 使用随机森林，输出特征重要性
    def importantFeaturesRF(self):
        clf = pre.randomForestClassifier()
        clf.fit(self.X_train, self.y_train)
        feat_labels = self.X.columns
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(self.X_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


    
if __name__ == "__main__":
    pre = Prediction()
    # regs = [('Logisitic RegressionCV',pre.logisiticRegression()),
    #         ('SVC',pre.SVM()),('Random Forest',pre.randomForestClassifier()),
    #         ('Decision Tree',pre.decisionTree()),
    #         ('Adaboost',pre.adaboostClassifier())]
    # estimators = [pre.logisiticRegression(), pre.SVM(), pre.decisionTree(), pre.randomForestClassifier(), pre.adaboostClassifier()]
    # pre.fit(regs)
    # pre.predict(regs)

    # 交叉验证
    # pre.crossScore(regs)
    # PCA和标准化
    # pre.pipline(estimators)

    
    # 决策树可视化
    # pre.viewTree()
    # f_old = open(OLD_PATH, 'r') 
    # f_new = open(NEW_PATH, 'w', encoding='utf-8') 
    # filename = 'dot_data_new.txt'
    # pre.font_conf(f_old,f_new, filename, TO_VIEW_PATH)

    # 随机森林可视化
    # pre.test_RF()

    pre.importantFeaturesRF()


