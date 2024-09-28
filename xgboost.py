import numpy as np  
import pandas as pd  
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, KFold 
from sklearn.model_selection import ShuffleSplit #分折交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from sklearn.model_selection import learning_curve
from joblib import dump, load


# Load your data and split it into training and testing sets
file_path = "datapreprocessing.xlsx"

data = pd.read_excel(file_path, sheet_name='pH') # change any sheet name to get the data
X=data.iloc[:,0:19].values
y=data.iloc[:,-1].values
print(y[:1])
print(X[:1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)


def _xgb_evaluate(n_estimators, learning_rate, max_depth, subsample, gamma, eta, colsample_bytree, min_child_weight,alpha):
    
    params = {
        'eval_metric': 'rmse',
        'n_estimators': int(n_estimators), 
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'subsample': subsample,
        'eta': eta,
        'gamma': gamma,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'alpha': alpha,
    }
    
    cv_result = xgb.cv(params, xgb.DMatrix(X_train, y_train), nfold=5)
   
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(_xgb_evaluate, {'n_estimators': (130,250), 
                                              'learning_rate': (0.06,0.08), 
                                              'max_depth': (4, 7),
                                              'subsample': (0.5, 0.7), 
                                              'gamma': (0,0.05), 
                                              'colsample_bytree': (0, 0.7),
                                              'min_child_weight': (0, 0.5),
                                              'eta': (0,0.1), 
                                              'alpha':(0,0.5) 
                                              }, random_state=40)

xgb_bo.maximize(init_points=10, n_iter=100) 

params_best =xgb_bo.max["params"]
score_best = xgb_bo.max["target"]

reg_val = XGBRegressor(n_estimators= int(params_best["n_estimators"]),
              learning_rate=params_best["learning_rate"], 
              max_depth=int(params_best["max_depth"]),
              subsample=params_best["subsample"],
              gamma= params_best["gamma"],
              colsample_bytree=params_best["colsample_bytree"],
              min_child_weight=params_best["min_child_weight"],
              eta=params_best["eta"],
              alpha=params_best["alpha"],
              random_state=40)
print('n_estimators:',int(params_best["n_estimators"]),
    'learning_rate=',params_best["learning_rate"],
    'max_depth=',int(params_best["max_depth"]),
    'subsample=',params_best["subsample"],
    'gamma=',params_best["gamma"],
    'colsample_bytree=',params_best["colsample_bytree"],
    'min_child_weight=',params_best["min_child_weight"],
    'eta=',params_best["eta"],
    'alpha=',params_best["alpha"])

best_model=reg_val.fit(X_train,y_train)
y_test_predict = best_model.predict(X_test)
y_train_predict = best_model.predict(X_train)
print('trainMSE:', mean_squared_error(y_train, y_train_predict))
print('trainRMSE:', np.sqrt(mean_squared_error(y_train, y_train_predict)))
print('trainMAE:', mean_absolute_error(y_train, y_train_predict))
print('testMSE:', mean_squared_error(y_test, y_test_predict))
print('testRMSE:', np.sqrt(mean_squared_error(y_test, y_test_predict)))
print('trainMAE:', mean_absolute_error(y_test, y_test_predict))
print('train-r2', r2_score(y_train, y_train_predict))
print('test-r2', r2_score(y_test, y_test_predict))
print(best_model.feature_importances_)

dump(best_model, 'model_XGBoost.joblib')