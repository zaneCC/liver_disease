
# 逻辑回归
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

# 逻辑回归
def logisiticRegression():
    lg =LogisticRegressionCV(multi_class="multinomial", solver="newton-cg")
    return lg

# 支持向量机
def SVM():
    # SMOTE_D-最优参数：C=8.736810045184805
    # SMOTE_Borderline1-最优参数：C=5415708639.990359
    # 随机过采样-最优参数：C=113722158.65263128
    # SMOTE - 最优参数：C=20.427613294134296
    sv =svm.SVC(C=8.736810045184805, probability=True)
    return sv

# TODO 运行前改参
# 决策树
def decisionTree():
    # 随机过采样-最优参数：'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 2
    # SMOTE-最优参数：{'criterion': 'gini', 'max_depth': 22, 'min_samples_split': 2}
    # SMOTE_Borderline1-最优参数：params={'criterion': 'gini', 'max_depth': 24, 'min_samples_split': 4}
    # SMOTE_D-最优参数：params={'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 6}

    dt = DecisionTreeClassifier(criterion='entropy' ,max_depth=9, min_samples_split=6)
    return dt

# TODO 运行前改参
# 随机森林
def randomForestClassifier():
    # 随机过采样-最优参数:   params={'rf_max_depth': 28, 'rf_n_estimators': 20, 'criterion': 'entropy'}
    # SMOTE-最优参数：            params={'rf_max_depth': 20, 'rf_n_estimators': 8, 'criterion': 'gini'}
    # SMOTE_Borderline1-最优参数：params={'rf_max_depth': 27, 'rf_n_estimators': 9, 'criterion': 'gini'}
    # SMOTE_D-最优参数：params={'rf_max_depth': 23, 'rf_n_estimators': 14, 'criterion': 'entropy'}
    return RandomForestClassifier(criterion='entropy', max_depth=23, n_estimators=14)

# TODO 运行前改参
# adaboost
def adaboostClassifier():
    return AdaBoostClassifier(algorithm='SAMME.R' ,learning_rate=0.6 ,n_estimators=60)

def xgboost():
    # SMOTE_D-最优参数：
    param = {'booster': 'gbtree', 'lambda': 0.016162497239068596, 'alpha': 0.005223776157313879,
              'subsample': 0.7118845749597382, 'colsample_bytree': 0.45835384576306376, 'max_depth': 7,
              'min_child_weight': 2, 'eta': 0.30301527084809826, 'gamma': 0.4401296769899615,
              'grow_policy': 'lossguide'}
    # SMOTE_Borderline1-最优参数：
    # param = {'booster': 'dart', 'lambda': 1.9358857695376388e-05, 'alpha': 0.00025477579048790544,
    #           'subsample': 0.8635832266574965, 'colsample_bytree': 0.963213639645865, 'max_depth': 9,
    #           'min_child_weight': 2, 'eta': 0.6167586940470704, 'gamma': 0.0014844363254459013,
    #           'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest',
    #           'rate_drop': 2.824588998714864e-07, 'skip_drop': 0.0033561299265673582}
    # 随机过采样-最优参数：
    # param = {'booster': 'dart', 'lambda': 0.00017552612672364366, 'alpha': 0.0004961649857205371,
    #           'subsample': 0.8566298691204985, 'colsample_bytree': 0.6106057323660206, 'max_depth': 9,
    #           'min_child_weight': 2, 'eta': 0.2544705495024535, 'gamma': 1.2517075201985975e-07,
    #           'grow_policy': 'lossguide', 'sample_type': 'uniform', 'normalize_type': 'forest',
    #           'rate_drop': 6.794465084467642e-07, 'skip_drop': 1.4585943922721661e-08}
    # SMOTE-最优参数：
    # param ={'booster': 'gbtree', 'lambda': 3.369992849615612e-08, 'alpha': 1.4450195138500373e-06,
    #  'subsample': 0.8901580563888312, 'colsample_bytree': 0.7594984417638513, 'max_depth': 9, 'min_child_weight': 2,
    #  'eta': 0.4538840438869842, 'gamma': 4.283265811681698e-06, 'grow_policy': 'depthwise'}

    return xgb.XGBClassifier(**param,use_label_encoder=False)

Logisitic_RegressionCV = 'Logisitic RegressionCV'
SVC = 'SVC'
Random_Forest = 'Random Forest'
Decision_Tree = 'Decision Tree'
XGBOOST = 'XGBoost'
MODELS_NAME = [Logisitic_RegressionCV, SVC, Random_Forest,Decision_Tree, XGBOOST]
