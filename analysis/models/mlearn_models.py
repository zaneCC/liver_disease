
# 逻辑回归
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import sys
sys.path.append(r'/Users/hear9000/Documents/codes/python_code/liver_disease/liver_disease/')
import config
import analysis.models.params as params
import analysis.models.model_constants as model_constants

CURRENT_PATH = config.PATH
CURRENT_TYPE = model_constants.TYPE_F1_PRE


# 逻辑回归
def logisiticRegression():
    lg =LogisticRegressionCV(multi_class="multinomial", solver="newton-cg")
    return lg

# 支持向量机
def SVM():
    # 随机过采样-最优参数：C=113722158.65263128               AUC:'svc_c': 7.884479277826943
    # SMOTE - 最优参数：C=20.427613294134296                AUC:'svc_c': 1.3122654687210835
    # SMOTE_Borderline1-最优参数：C=5415708639.990359       AUC:'svc_c': 9976509030.539387
    # SMOTE_D-最优参数：C=8.736810045184805                 AUC:'svc_c': 1.6040637815548981
    param = params.get_svm_params(CURRENT_PATH, CURRENT_TYPE)
    sv =svm.SVC(**param, probability=True)
    return sv

# TODO 运行前改参
# 决策树
def decisionTree():
    # 随机过采样-最优参数：
    # param={'criterion': 'entropy', 'max_depth': 12, 'min_samples_split': 2}
    #   AUC:
    # param={'criterion': 'gini', 'max_depth': 32, 'min_samples_split': 8}

    # SMOTE-最优参数：
    # param={'criterion': 'gini', 'max_depth': 22, 'min_samples_split': 2}
    #   AUC:
    # param={'criterion': 'gini', 'max_depth': 16, 'min_samples_split': 4}

    # SMOTE_Borderline1-最优参数：
    # param={'criterion': 'gini', 'max_depth': 24, 'min_samples_split': 4}
    #   AUC:
    # param={'criterion': 'gini', 'max_depth': 28, 'min_samples_split': 2}

    # SMOTE_D-最优参数：
    # param={'criterion': 'entropy', 'max_depth': 9, 'min_samples_split': 6}
    #   AUC:
    # param={'criterion': 'entropy', 'max_depth': 21, 'min_samples_split': 12}
    param = params.get_dt_params(CURRENT_PATH, CURRENT_TYPE)
    dt = DecisionTreeClassifier(**param)
    return dt

# TODO 运行前改参
# 随机森林
def randomForestClassifier():
    # 随机过采样-最优参数:   params={'rf_max_depth': 28, 'rf_n_estimators': 20, 'criterion': 'entropy'}
    #                         AUC:
    # params={'max_depth': 21, 'n_estimators': 13, 'criterion': 'gini'}

    # SMOTE-最优参数：            params={'max_depth': 20, 'n_estimators': 8, 'criterion': 'gini'}
    #                         AUC:
    # params={'max_depth': 21, 'n_estimators': 19, 'criterion': 'entropy'}

    # SMOTE_Borderline1-最优参数：params={'max_depth': 27, 'n_estimators': 9, 'criterion': 'gini'}
    #                         AUC:
    # params={'max_depth': 30, 'n_estimators': 10, 'criterion': 'gini'}

    # SMOTE_D-最优参数：params={'max_depth': 23, 'n_estimators': 14, 'criterion': 'entropy'}
    #                         AUC:
    # params={'max_depth': 32, 'n_estimators': 17, 'criterion': 'entropy'}
    param = params.get_rf_params(CURRENT_PATH, CURRENT_TYPE)
    return RandomForestClassifier(**param)

# TODO 运行前改参
# adaboost
def adaboostClassifier():
    return AdaBoostClassifier(algorithm='SAMME.R' ,learning_rate=0.6 ,n_estimators=60)

def xgboost():
    # 随机过采样-最优参数：
    # param = {'booster': 'dart', 'lambda': 0.00017552612672364366, 'alpha': 0.0004961649857205371,
    #           'subsample': 0.8566298691204985, 'colsample_bytree': 0.6106057323660206, 'max_depth': 9,
    #           'min_child_weight': 2, 'eta': 0.2544705495024535, 'gamma': 1.2517075201985975e-07,
    #           'grow_policy': 'lossguide', 'sample_type': 'uniform', 'normalize_type': 'forest',
    #           'rate_drop': 6.794465084467642e-07, 'skip_drop': 1.4585943922721661e-08}
    # AUC:
    # param={'booster': 'dart', 'lambda': 5.712025830112571e-08, 'alpha': 0.021184676003902867,
    # 'subsample': 0.9015855999686154, 'colsample_bytree': 0.24374656182912582, 'max_depth': 7,
    # 'min_child_weight': 2, 'eta': 0.9952157917931137, 'gamma': 0.0014989524392155474,
    # 'grow_policy': 'depthwise', 'sample_type': 'weighted', 'normalize_type': 'tree',
    # 'rate_drop': 8.68809834753798e-06, 'skip_drop': 1.3620523416984143e-07}

    # SMOTE-最优参数：
    # param ={'booster': 'gbtree', 'lambda': 3.369992849615612e-08, 'alpha': 1.4450195138500373e-06,
    #  'subsample': 0.8901580563888312, 'colsample_bytree': 0.7594984417638513, 'max_depth': 9, 'min_child_weight': 2,
    #  'eta': 0.4538840438869842, 'gamma': 4.283265811681698e-06, 'grow_policy': 'depthwise'}
    # AUC:
    # param = {'booster': 'gblinear', 'lambda': 1.2981643589951566e-08, 'alpha': 0.025060444875515935,
    #          'subsample': 0.8144462739426666, 'colsample_bytree': 0.502435521745772}


    # SMOTE_Borderline1-最优参数：
    # param = {'booster': 'dart', 'lambda': 1.9358857695376388e-05, 'alpha': 0.00025477579048790544,
    #           'subsample': 0.8635832266574965, 'colsample_bytree': 0.963213639645865, 'max_depth': 9,
    #           'min_child_weight': 2, 'eta': 0.6167586940470704, 'gamma': 0.0014844363254459013,
    #           'grow_policy': 'depthwise', 'sample_type': 'uniform', 'normalize_type': 'forest',
    #           'rate_drop': 2.824588998714864e-07, 'skip_drop': 0.0033561299265673582}
    # AUC:
    # param={'booster': 'gbtree', 'lambda': 8.832863215352322e-05, 'alpha': 0.0029490031472444896,
    # 'subsample': 0.9790727032229547, 'colsample_bytree': 0.8129539552905347, 'max_depth': 9,
    # 'min_child_weight': 6, 'eta': 0.5546283949833964, 'gamma': 0.00018408651570821504, 'grow_policy': 'lossguide'}
    #
    # SMOTE_D-最优参数：
    # param = {'booster': 'gbtree', 'lambda': 0.016162497239068596, 'alpha': 0.005223776157313879,
    #           'subsample': 0.7118845749597382, 'colsample_bytree': 0.45835384576306376, 'max_depth': 7,
    #           'min_child_weight': 2, 'eta': 0.30301527084809826, 'gamma': 0.4401296769899615,
    #           'grow_policy': 'lossguide'}
    # AUC:
    # param={'booster': 'gbtree', 'lambda': 6.3613063490166845e-06, 'alpha': 0.0012803748439886838,
    # 'subsample': 0.9475524177321181, 'colsample_bytree': 0.983958990723794, 'max_depth': 7,
    # 'min_child_weight': 5, 'eta': 0.9928978687905468, 'gamma': 0.0008203572059634942, 'grow_policy': 'lossguide'}

    param = params.get_xgboost_params(CURRENT_PATH,CURRENT_TYPE)
    return xgb.XGBClassifier(**param,use_label_encoder=False)

Logisitic_RegressionCV = 'Logisitic RegressionCV'
SVC = 'SVC'
Random_Forest = 'Random Forest'
Decision_Tree = 'Decision Tree'
XGBOOST = 'XGBoost'
MODELS_NAME = [Logisitic_RegressionCV, SVC, Random_Forest,Decision_Tree, XGBOOST]
