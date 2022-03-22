'''
    肝病预测
'''
OS_MAC_PATH = '/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease'
OS_WINDOWS_PATH = 'E:/liver_disease/liver_disease'

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(r'/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/')

import constants,config
import analysis.models.mlearn_models as models
import utils.misc as misc
import utils.data_utils as data_utils
from sklearn.model_selection import train_test_split, validation_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



class Prediction():
    
    # 加载数据
    def initData(self,path=config.PATH):
        df = pd.read_csv(path)
        cols = df.columns.values.tolist()
        # cols.remove('Unnamed: 0')
        # cols.remove('INHOSPTIAL_ID')
        cols.remove('ZHENGHOU1')

        self.X = df[cols] 
        self.y = df['ZHENGHOU1']

        self.split()

    def __init__(self):
        self.misc = misc.Misc()
        self.isShow = True # 是否展示图像
        self.initData()
    
    def split(self):
        # TODO 使用 StratifiedKFold 拆分数据
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y,test_size = 0.2,random_state=0)
        ss = StandardScaler()
        self.X_train = ss.fit_transform(X_train)
        self.X_test = ss.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def fit(self,regs):
        for key,(name, i) in enumerate(regs):
            i.fit(self.X_train,self.y_train)
        
    def predict(self,regs):
        k = 10
        sfolder = StratifiedKFold(n_splits=k, shuffle=True, random_state=None)
        _x = self.X.to_numpy()
        # _y = self.y.to_numpy()
        _y = data_utils.unify_lable(self.y)

        for key, (name, i) in enumerate(regs):
            # 获得 10 次平均结果：精度、准确度、召回率、f1值
            aver_f1, aver_precision, var_precision_0, var_precision_1, var_f1_0, var_f1_1 = [], [], [], [], [], []
            aver_recall,var_recall_0,var_recall_1 = [],[],[]
            # 方差
            std_f1, std_precision = [],[]

            for train_index, test_index in sfolder.split(_x, _y):
                X_train, X_test = _x[train_index], _x[test_index]
                y_train, y_test = _y[train_index], _y[test_index]

                i.fit(X_train, y_train)
                pre = i.predict(X_test)

                # 精度
                precision = precision_score(y_test, pre, average=None)
                var_precision_0.append(precision[0])
                var_precision_1.append(precision[1])
                # f1
                f1 = f1_score(y_test, pre, average=None)
                var_f1_0.append(f1[0])
                var_f1_1.append(f1[1])
                # 召回率
                recall = recall_score(y_test, pre, average=None)
                var_recall_0.append(recall[0])
                var_recall_1.append(recall[1])

            aver_precision.append(np.mean(var_precision_0))
            aver_precision.append(np.mean(var_precision_1))
            aver_f1.append(np.mean(var_f1_0))
            aver_f1.append(np.mean(var_f1_1))
            aver_recall.append(np.mean(var_recall_0))
            aver_recall.append(np.mean(var_recall_1))
            std_f1.append(np.std(var_f1_0))
            std_f1.append(np.std(var_f1_1))
            std_precision.append(np.std(var_precision_0))
            std_precision.append(np.std(var_precision_1))

            print('模型:', name)
            print('平均精度：', aver_precision, ', std：', std_precision)
            print('平均 f1：', aver_f1, ', std：', std_f1)
            print('平均召回率：', aver_recall)


            # TODO 保存结果

            
    ''' 
        求均值、标准差
    '''
    def predictAverage(self,regs, k=6): # 6次采样，10次交叉验证
        # 先在work_2021615 中执行 resampling，生成样本
        sfolder = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        _x = self.X.to_numpy()
        _y = data_utils.unify_lable(self.y)

        for key, (name, i) in enumerate(regs):
            # 获得 10 次平均结果：精度、准确度、召回率、f1值
            aver_f1, aver_precision, var_precision_0, var_precision_1, var_f1_0, var_f1_1 = [], [], [], [], [], []
            aver_recall, var_recall_0, var_recall_1 = [], [], []
            # 方差
            std_f1, std_precision = [], []
            for j in range(k): # k 次采样
                print('--------------- 第', j, '次采样 ------------------')
                path = config.config_path_compare_test(j) # 修改数据
                self.initData(path)

                for train_index, test_index in sfolder.split(_x, _y):
                    X_train, X_test = _x[train_index], _x[test_index]
                    y_train, y_test = _y[train_index], _y[test_index]

                    i.fit(X_train, y_train)
                    pre = i.predict(X_test)

                    # 精度
                    precision = precision_score(y_test, pre, average=None)
                    var_precision_0.append(precision[0])
                    var_precision_1.append(precision[1])
                    # f1
                    f1 = f1_score(y_test, pre, average=None)
                    var_f1_0.append(f1[0])
                    var_f1_1.append(f1[1])
                    # 召回率
                    recall = recall_score(y_test, pre, average=None)
                    var_recall_0.append(recall[0])
                    var_recall_1.append(recall[1])

            aver_precision.append(np.mean(var_precision_0))
            aver_precision.append(np.mean(var_precision_1))
            aver_f1.append(np.mean(var_f1_0))
            aver_f1.append(np.mean(var_f1_1))
            aver_recall.append(np.mean(var_recall_0))
            aver_recall.append(np.mean(var_recall_1))
            std_f1.append(np.std(var_f1_0))
            std_f1.append(np.std(var_f1_1))
            std_precision.append(np.std(var_precision_0))
            std_precision.append(np.std(var_precision_1))

            print('模型:', name)
            print('平均精度：', aver_precision, ', std：', std_precision)
            print('平均 f1：', aver_f1, ', std：', std_f1)
            print('平均召回率：', aver_recall)




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

    def roc_k(self, regs, k=6):
        aver_auc = {}
        for key, (name, i) in enumerate(regs):
            aver = 0
            for j in range(k):
                # RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + '实验' + '/对比实验/' + str(
                #     j) + '/随机过采样-采样-汇总表.csv'
                # SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE-采样-汇总表.csv'
                # SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_Borderline1-采样-汇总表.csv'
                # SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_D-采样-汇总表.csv'
                # SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/'+ '实验' +'/对比实验/'+ str(j) +'/SMOTE_Borderline_D-采样-汇总表.csv'

                ''' 修改论文数据的时候使用 '''
                RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
                    j) + '/随机过采样-采样-汇总表.csv'
                SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
                    j) + '/SMOTE-采样-汇总表.csv'
                SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
                    j) + '/SMOTE_Borderline1-采样-汇总表.csv'
                SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
                    j) + '/SMOTE_D-采样-汇总表.csv'
                SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/对比实验/' + str(
                    j) + '/SMOTE_Borderline_D-采样-汇总表.csv'

                # 舌象-证，数据
                # if config.IS_SHE:
                #     RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
                #         j) + '/随机过采样-采样-汇总表.csv'
                #     SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
                #         j) + '/SMOTE-采样-汇总表.csv'
                #     SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
                #         j) + '/SMOTE_Borderline1-采样-汇总表.csv'
                #     SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
                #         j) + '/SMOTE_D-采样-汇总表.csv'
                #     SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/舌象/对比实验/' + str(
                #         j) + '/SMOTE_Borderline_D-采样-汇总表.csv'

                # 舌象-证，数据
                ''' 修改论文数据的时候使用 '''
                if config.IS_SHE:
                    RANDOM_OVER_SAMPLER_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
                        j) + '/随机过采样-采样-汇总表.csv'
                    SMOTE_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
                        j) + '/SMOTE-采样-汇总表.csv'
                    SMOTE_BORDERLINE1_COMPARE_MERGE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
                        j) + '/SMOTE_Borderline1-采样-汇总表.csv'
                    SMOTE_D_MERGE_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
                        j) + '/SMOTE_D-采样-汇总表.csv'
                    SMOTE_Borderline_D_COMPARE_CSV_PATH = constants.OS_PATH + '/output/SMOTE/' + constants.PAPER_VERSION + '/症状_舌象/对比实验/' + str(
                        j) + '/SMOTE_Borderline_D-采样-汇总表.csv'

                path = SMOTE_Borderline_D_COMPARE_CSV_PATH
                self.initData(path)

                if name == 'Random Forest' or name == 'Decision Tree':  # 这两个分类器没有 decision_function 方法
                    y_pred_proba = i.fit(self.X_train, self.y_train).predict_proba(self.X_test)
                    fpr, tpr, threshold = roc_curve(self.y_test, y_pred_proba[:, 1], pos_label=2)  # 计算真正率和假正率
                else:
                    y_pred_proba = i.fit(self.X_train, self.y_train).decision_function(self.X_test)
                    fpr, tpr, threshold = roc_curve(self.y_test, y_pred_proba, pos_label=2)  # 计算真正率和假正率

                roc_auc = auc(fpr, tpr)
                aver += roc_auc
            aver_auc[name] = aver / k
        print(aver_auc)

    def roc(self,regs):
        self.misc.figure()
        color = ['r','g','b','c','y']
        for key,(name, i) in enumerate(regs):
            if name == 'Random Forest' or name == 'Decision Tree':  # 这两个分类器没有 decision_function 方法
                y_pred_proba = i.fit(self.X_train, self.y_train).predict_proba(self.X_test)
                fpr,tpr,threshold = roc_curve(self.y_test, y_pred_proba[:,1], pos_label=2) # 计算真正率和假正率
            else:
                y_pred_proba = i.fit(self.X_train, self.y_train).decision_function(self.X_test)
                fpr,tpr,threshold = roc_curve(self.y_test, y_pred_proba, pos_label=2) # 计算真正率和假正率

            # 汇总表 - 舌象: MERGE_CSV_DIA_TONGUE_PATH
            # 随机过采样: TUE_RANDOM_OVER_SAMPLER_CSV_PATH
            # SMOTE: TUE_SMOTE_MERGE_CSV_PATH
            # SMOTE
            # Borderline1: TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH
            # SMOTE_D: TUE_SMOTE_D_MERGE_CSV_PATH
            # SMOTE_BORDERLINE_D: TUE_SMOTE_Borderline_D_CSV_PATH

            roc_auc = auc(fpr,tpr)

            # 绘图
            title = '原始数据\'s ROC'
            if config.PATH == constants.SMOTE_MERGE_CSV_PATH or config.PATH == constants.TUE_SMOTE_MERGE_CSV_PATH:
                title = 'SMOTE \'s ROC'
            elif config.PATH == constants.SMOTE_BORDERLINE1_MERGE_CSV_PATH or config.PATH == constants.TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH:
                title = 'SMOTE Borderline\'s ROC'
            elif config.PATH == constants.SMOTE_D_MERGE_CSV_PATH or config.PATH == constants.TUE_SMOTE_D_MERGE_CSV_PATH:
                title = 'SMOTE D\'s ROC'
            elif config.PATH == constants.SMOTE_Borderline_D_CSV_PATH or config.PATH == constants.TUE_SMOTE_Borderline_D_CSV_PATH:
                title = 'SMOTE Borderline D\'s ROC'
            elif config.PATH == constants.RANDOM_OVER_SAMPLER_CSV_PATH or config.PATH == constants.TUE_RANDOM_OVER_SAMPLER_CSV_PATH:
                title = 'RANDOM OVER SAMPLER\'s ROC'
            self.misc.plotAUC(fpr, tpr, roc_auc, color=color[key], label=name, title = title)

        self.misc.show()




if __name__ == "__main__":
    pre = Prediction()

    # 预测
    regs = [
            (models.Logisitic_RegressionCV,models.logisiticRegression()),
            (models.SVC,models.SVM()),
            (models.Random_Forest,models.randomForestClassifier()),
            (models.Decision_Tree,models.decisionTree()),
            # ('Adaboost',models.adaboostClassifier()),
            (models.XGBOOST, models.xgboost())
            # ('NB', pre.NB())
            ]
    # estimators = [pre.logisiticRegression(), pre.SVM(), pre.decisionTree(), pre.randomForestClassifier(), pre.adaboostClassifier()]
    # pre.fit(regs)
    # pre.predict(regs)
    pre.predictAverage(regs)

    # 交叉验证
    # pre.crossScore(regs)
    # PCA和标准化
    # pre.pipline(estimators)

    # pre.roc(regs)
    # pre.roc_k(regs)

    # 决策树可视化
    # pre.viewTree()
    # f_old = open(OLD_PATH, 'r')
    # f_new = open(NEW_PATH, 'w', encoding='utf-8')
    # filename = 'dot_data_new.txt'
    # pre.font_conf(f_old,f_new, NEW_PATH, TO_VIEW_PATH)

    # 随机森林可视化
    # pre.test_RF()

    # 输出特征重要性
    # pre.importantFeaturesRF()


