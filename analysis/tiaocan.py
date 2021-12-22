import numpy as np
import pandas as pd
import sys
# sys.path.append(r'/Users/zhouzhan/Documents/codes/python_code/liver_disease/liver_disease/')
sys.path.append(r'E:/liver_disease/liver_disease')
import constants
import utils.misc as misc
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn.tree as tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import roc_curve, auc

'''
汇总表-未做特征选择:    MERGE_CSV_PATH
汇总表-特征选择后:     SELECTION_MERGE_CSV_PATH
人工选择特征:         SYMP_MAIN_ACC_DIAGNOSIS_PATH
SMOTE:              ANALYSIS_SMOTE_MERGE_CSV_PATH
SMOTE Borderline1:  ANALYSIS_SMOTE_BORDERLINE1_MERGE_CSV_PATH
SMOTE_D:            ANALYSIS_SMOTE_D_MERGE_CSV_PATH
SMOTE_BORDERLINE_D: ANALYSIS_SMOTE_Borderline_D_CSV_PATH
随机过采样:           ANALYSIS_RANDOM_OVER_SAMPLER_CSV_PATH
'''
PATH = constants.ANALYSIS_SMOTE_Borderline_D_CSV_PATH

misc = misc.Misc()
df = pd.read_csv(PATH)

cols = df.columns.values.tolist()
# cols.remove('INHOSPTIAL_ID')
cols.remove('ZHENGHOU1')

X = df[cols] 
y = df['ZHENGHOU1']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

# 查看诊断结果的分布情况，看是否存在数据不均衡

print('证的分布情况:',Counter(df['ZHENGHOU1']))

# 调参-网格搜索-决策树
param_grid = {'criterion':['entropy','gini'],
             'max_depth':[3,4,5,6,7],
             'min_samples_split':[2,4,8,10,40,50]
             }

# # 调参-网格搜索-随机森林
# param_grid2 = {
#     'criterion':['entropy','gini'],
#     'max_depth':[16,20,24,30],
#     'n_estimators':[20,30,40,50],
#     'max_features':[0.2,0.4,0.6],
#     'min_samples_split':[2,4,6],
# }
# # 调参-网格搜索-Adaboost
# param_grid3 = {
#     'algorithm':['SAMME','SAMME.R'],
#     'n_estimators':[60,80,100,120],
#     'learning_rate':[0.6,0.8,1],
# }

# rfc = RandomForestClassifier()
clf = tree.DecisionTreeClassifier() 
# adaboost = AdaBoostClassifier()

# fit = clf.fit(X_train,y_train)
# test_result = clf.predict(X_test)

clfcv = GridSearchCV(estimator=clf,param_grid=param_grid,cv=4,scoring='roc_auc') # ,scoring='roc_auc' 
fit = clfcv.fit(X_train,y_train)

test_result = clfcv.predict(X_test)
# 模型评估
import sklearn.metrics as metrics
print('准确度：')
print(metrics.classification_report(y_test,test_result))
# 画出 ROC
misc.figure()
y_pred_proba = fit.predict_proba(X_test)
fpr,tpr,threshold = roc_curve(y_test, y_pred_proba[:,1], pos_label=2) # 计算真正率和假正率
roc_auc = auc(fpr,tpr)
misc.plotAUC(fpr, tpr, roc_auc, color='r', label='ROC', title = 'ROC')
misc.show()

# 网格搜索最优参数
print(clfcv.best_params_)
