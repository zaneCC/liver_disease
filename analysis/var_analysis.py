import numpy as np
import pandas as pd
import sys
sys.path.append(r'/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/')
# sys.path.append(r'E:/liver_disease/liver_disease')
import constants,config
import analysis.models.mlearn_models as models
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import matplotlib.ticker as ticker

''' 症状-证
汇总表-未做特征选择:    MERGE_CSV_PATH
汇总表-特征选择后:     SELECTION_MERGE_CSV_PATH
人工选择特征:         SYMP_MAIN_ACC_DIAGNOSIS_PATH

SMOTE:              ANALYSIS_SMOTE_MERGE_CSV_PATH
SMOTE_Borderline1:  ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH
SMOTE_D:            ANALYSIS_SMOTE_D_MERGE_CSV_PATH
SMOTE_BORDERLINE_D: ANALYSIS_SMOTE_Borderline_D_CSV_PATH
随机过采样:           ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH
'''
''' 舌象-证
汇总表-舌象:          MERGE_CSV_DIA_TONGUE_PATH
随机过采样:           TUE_RANDOM_OVER_SAMPLER_CSV_PATH
SMOTE:              TUE_SMOTE_MERGE_CSV_PATH
SMOTE Borderline1:  TUE_SMOTE_BORDERLINE1_MERGE_CSV_PATH
SMOTE_D:            TUE_SMOTE_D_MERGE_CSV_PATH
SMOTE_BORDERLINE_D: TUE_SMOTE_Borderline_D_CSV_PATH
'''
PATH = constants.TUE_SMOTE_MERGE_CSV_PATH


class VarAnalysis():

    def __init__(self):
        # 设置字体
        # mpl.rcParams['font.sans-serif'] = [u'simHei']
        # mpl.rcParams['axes.unicode_minus'] = False

        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        # 读数据
        self.read_data()
        
    def read_data(self,path=PATH):
        self.df = pd.read_csv(path)
        cols = self.df.columns.values.tolist()
        # cols.remove('INHOSPTIAL_ID')
        cols.remove('ZHENGHOU1')

        X = self.df[cols] 
        y = self.df['ZHENGHOU1']
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
        self.path = path

    # 决策树
    def get_DTC_important_var(self,top=20, k=50):
        # 随机过采样-最优参数：          {'criterion': 'gini', 'max_depth': 15, 'min_samples_split': 2}   AUC:0.81
        # SMOTE-最优参数：             {'criterion': 'entropy', 'max_depth': 14, 'min_samples_split': 6}   AUC:0.82
        # SMOTE_Borderline1-最优参数： {'criterion': 'entropy', 'max_depth': 20, 'min_samples_split': 6} AUC:0.83
        # SMOTE_D-最优参数：           {'criterion': 'gini', 'max_depth': 17, 'min_samples_split': 10}     AUC:0.86
        # SMOTE_BORDERLINE_D-最优参数：{'criterion': 'gini', 'max_depth': 18, 'min_samples_split': 10}  AUC:0.82

        # 根据样本配置最优参数
        # if self.path == constants.ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH: # 随机过采样
        #     criterion = 'gini';max_depth = 15;min_samples_split = 2
        # elif self.path == constants.ANALYSIS_SMOTE_MERGE_CSV_PATH:      # SMOTE
        #     criterion = 'entropy';max_depth = 14;min_samples_split = 6
        # elif self.path == constants.ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH:  # SMOTE_Borderline1
        #     criterion = 'entropy';max_depth = 20;min_samples_split = 6
        # elif self.path == constants.ANALYSIS_SMOTE_D_MERGE_CSV_PATH:  # SMOTE_D
        #     criterion = 'gini';max_depth = 18;min_samples_split = 10
        # elif self.path == constants.ANALYSIS_SMOTE_Borderline_D_CSV_PATH:  # SMOTE_BORDERLINE_D
        #     criterion = 'gini';max_depth = 18;min_samples_split = 10

        importances = np.zeros(len(self.X_test.columns))
        s = 0
        for i in range(k):
            # clf = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split)
            clf = models.decisionTree()
            clf.fit(self.X_train,self.y_train)
            test_result = clf.predict(self.X_test)
            importances += clf.feature_importances_
            # print(clf.feature_importances_.sum())

        importances = importances / k
        # print(importances)
        # 模型评估
        print('决策树准确度：')
        print(metrics.classification_report(self.y_test,test_result))

        features = list(self.X_test.columns)
        indices = np.argsort(importances)[::-1]
        top_indices = indices[0:top]
        print(top_indices)
        top_features = len(top_indices)
        return top_features, importances, top_indices, features

    # 随机森林
    def get_RFC_important_var(self, top=20, k=50):
        # 随机过采样-最优参数:         {'criterion': 'entropy', 'max_depth': 20, 'max_features': 0.2, 'min_samples_split': 2, 'n_estimators': 40}    AUC:0.82
        # SMOTE-最优参数：            {'criterion': 'entropy', 'max_depth': 12, 'max_features': 0.4, 'min_samples_split': 4, 'n_estimators': 16}    AUC:0.80
        # SMOTE_Borderline1-最优参数：{'criterion': 'gini', 'max_depth': 13, 'max_features': 0.2, 'min_samples_split': 2, 'n_estimators': 14}   AUC:0.83
        # SMOTE_D-最优参数：          {'criterion': 'gini', 'max_depth': 13, 'max_features': 0.2, 'min_samples_split': 2, 'n_estimators': 12} AUC:0.86
        # SMOTE_BORDERLINE_D-最优参数:{'criterion': 'entropy', 'max_depth': 14, 'max_features': 0.6, 'min_samples_split': 2, 'n_estimators': 20}   AUC:0.85

        # 根据样本配置最优参数
        # if self.path == constants.ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH: # 随机过采样
        #     criterion = 'entropy';max_depth = 20;max_features=0.2;min_samples_split = 2;n_estimators=40
        # elif self.path == constants.ANALYSIS_SMOTE_MERGE_CSV_PATH:      # SMOTE
        #     criterion = 'entropy';max_depth = 12;max_features=0.4;min_samples_split = 4;n_estimators=16
        # elif self.path == constants.ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH:  # SMOTE_Borderline1
        #     criterion = 'gini';max_depth = 13;max_features=0.2;min_samples_split = 2;n_estimators=14
        # elif self.path == constants.ANALYSIS_SMOTE_D_MERGE_CSV_PATH:  # SMOTE_D
        #     criterion = 'gini';max_depth = 13;max_features=0.2;min_samples_split = 2;n_estimators=12
        # elif self.path == constants.ANALYSIS_SMOTE_Borderline_D_CSV_PATH:  # SMOTE_BORDERLINE_D
        #     criterion = 'entropy';max_depth = 14;max_features=0.6;min_samples_split = 2;n_estimators=20

        importances = np.zeros(len(self.X_test.columns))

        for i in range(k):
            # rfc = RandomForestClassifier(criterion='entropy',max_depth=7,max_features=0.6,min_samples_split=8,n_estimators=20)
            rfc = models.randomForestClassifier()
            rfc.fit(self.X_train,self.y_train)
            # test_result = rfc.predict(self.X_test)
            importances += rfc.feature_importances_
            # 模型评估
            # print('随机森林准确度：')
            # print(metrics.classification_report(self.y_test,test_result))

        importances = importances / k
        print(importances)

        features = list(self.X_test.columns)
        indices = np.argsort(importances)[::-1]
        top_indices = indices[0:top]
        print(top_indices)
        top_features = len(top_indices)
        return top_features, importances, top_indices, features
        
    def get_path(self,title,type=0):
        p_path = ''
        t_path = ''
        if type == 0:
            title = '决策树-' + title
        elif type == 1:
            title = '随机森林-' + title

        if config.IS_SHE:
            os_path = constants.OS_PATH + '/output/特征贡献度/舌象'
        else:
            os_path = constants.OS_PATH + '/output/特征贡献度'

        if self.path == constants.ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH or self.path == constants.TUE_ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH: # 随机过采样
            p_path = os_path + '/随机过采样/'+ title + '.png'
            t_path = os_path + '/随机过采样/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_MERGE_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_MERGE_CSV_PATH: # SMOTE
            p_path = os_path + '/SMOTE/'+ title +'.png'
            t_path = os_path + '/SMOTE/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH: # SMOTE Borderline1
            p_path = os_path + '/SMOTE_BORDERLINE/'+ title +'.png'
            t_path = os_path + '/SMOTE_BORDERLINE/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_D_MERGE_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_D_MERGE_CSV_PATH: # SMOTE_D
            p_path = os_path + '/SMOTE_D/'+ title +'.png'
            t_path = os_path + '/SMOTE_D/'+ title + '.txt'
        elif self.path == constants.ANALYSIS_SMOTE_Borderline_D_CSV_PATH or self.path == constants.TUE_ANALYSIS_SMOTE_Borderline_D_CSV_PATH: # SMOTE_BORDERLINE_D
            p_path = os_path + '/SMOTE_BORDERLINE_D/'+ title +'.png'
            t_path = os_path + '/SMOTE_BORDERLINE_D/'+ title + '.txt'
        return p_path, t_path

    def show_important_var(self,top_features,importances,top_indices,features, key, type,title='特征贡献度'):
        # plt.figure()
        ax = plt.subplot(2,2,key+1)
        ax.set_title(title)

        # plt.title(title)
        plt.bar(range(top_features), importances[top_indices], color="g", align="center")
        plt.xticks(range(top_features), [features[i] for i in top_indices], rotation='90')
        plt.xlim([-1, top_features])

        # plt.tick_params(axis='x', which='major', labelsize=8)
        # plt.tight_layout()

        p_path, t_path = self.get_path(title,type)
        # 保存图片
        plt.savefig(p_path)
        # plt.show()

        # 写入文件-各个特征的重要度
        try:
            f = open(t_path, 'w')
            for i in top_indices:
                s = "{0} - {1:.3f}".format(features[i], importances[i]/5)
                f.write(s+'\n')
        except FileNotFoundError:
            os.mknod(t_path)

    def show(self):
        plt.tight_layout()
        plt.show()
        
    # 所有采样方法的平均重要度特征排名-前top位-写入文件
    def get_top_important_var_by_file(self,regs,value,top=10):
        # 初始化计数字典
        dic = {}
        for i in self.X_test.columns:
            dic[i] = 0
        if value == '决策树':
            type = 0
        elif value == '随机森林':
            type = 1
        # 迭代所有算法采样后特征重要性文件
        for key,(title, path) in enumerate(regs):
            self.read_data(path)
            p_path, t_path = self.get_path(title, type)

            with open(t_path) as f:
                for line1 in f:
                    line1 = line1.strip()
                    strlist = line1.split(' - ')
                    strlist[1] = float(strlist[1])
                    dic[strlist[0]] = dic[strlist[0]] + strlist[1]
        # 获得 top 重要特征
        sorted_dic = sorted(dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)[0:top]

        keys = []
        values = []
        for i in sorted_dic:
            keys.append(i[0])
            values.append(i[1])

        # 展示
        plt.figure()
        plt.title('所有采样-'+value+'-特征重要性top'+str(top))
        plt.bar(range(len(keys)), values, color="g", align="center")
        plt.xticks(range(len(keys)), keys, rotation='90')
        # plt.xlim([-1, keys])

        # 写入文件
        path = constants.OS_PATH + '/output/特征贡献度/所有采样-'+value+'-特征重要性top'+str(top)+'.txt'
        try:
            f = open(path, 'w')
            r = 0
            for s in sorted_dic[0:10]:
                r += s[1]
                f.write(str(s)+'\n')
            f.write("总贡献度："+str(r)+'\n')
        except FileNotFoundError:
            os.mknod(path)
        
if __name__ == '__main__':
    va = VarAnalysis()
    
    # 决策树    随机森林
    value = '随机森林'
    if config.IS_SHE:
        regs = [
            ('随机过采样', constants.TUE_ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH),
            ('SMOTE', constants.TUE_ANALYSIS_SMOTE_MERGE_CSV_PATH),
            ('SMOTE_Borderline', constants.TUE_ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH),
            ('SMOTE_D', constants.TUE_ANALYSIS_SMOTE_D_MERGE_CSV_PATH)
            # ('SMOTE_BORDERLINE_D', constants.TUE_ANALYSIS_SMOTE_Borderline_D_CSV_PATH),
        ]
    else:
        regs = [
                ('随机过采样',constants.ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH),
                ('SMOTE',constants.ANALYSIS_SMOTE_MERGE_CSV_PATH),
                ('SMOTE_Borderline',constants.ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH),
                ('SMOTE_D',constants.ANALYSIS_SMOTE_D_MERGE_CSV_PATH)
                # ('SMOTE_BORDERLINE_D',constants.ANALYSIS_SMOTE_Borderline_D_CSV_PATH),
                ]

    # for key,(name, path) in enumerate(regs):
    #     va.read_data(path)
    #     # 生成特征贡献度文件
    #     # 决策树
    #     if value == '决策树':
    #         top_features, importances, top_indices, features = va.get_DTC_important_var()
    #         va.show_important_var(top_features, importances, top_indices, features, key, type=0,title=name)
    #     elif value == '随机森林':
    #         top_features, importances, top_indices, features = va.get_RFC_important_var()
    #         va.show_important_var(top_features, importances, top_indices, features, key, type=1,title=name)
    #
    # va.show()

    # 获得所有样本重要特征
    va.get_top_important_var_by_file(regs=regs,value=value)
    va.show()

    # top_features, importances, top_indices, features = va.get_RFC_important_var()
    # va.show_important_var(top_features, importances, top_indices, features, title='特征贡献度-随机森林-随机过采样')