import numpy as np  
import pandas as pd  
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from bayes_opt import BayesianOptimization
from time import time
from sklearn.model_selection import learning_curve
from joblib import dump, load


file_path = "datapreprocessing.xlsx"
data = pd.read_excel(file_path, sheet_name='pH') # change any sheet name to get the data
X=data.iloc[:,0:19].values
y=data.iloc[:,-1].values
print(X[:1])
print(y[:1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=119)
xg_train = xgb.DMatrix(X_train, label=y_train)
Xg_test = xgb.DMatrix(X_test)

def _rf_evaluate(n_estimators, max_features, max_depth,min_samples_leaf,min_samples_split): 
    reg = RandomForestRegressor(n_estimators= int(n_estimators), 
                                max_features= int(max_features),
                                max_depth=int(max_depth),
                                min_samples_leaf=int(min_samples_leaf),
                                min_samples_split=int(min_samples_split),
                                bootstrap=True,
                                random_state=90)
    
    kf = KFold(n_splits=5)
    scores = cross_val_score(reg, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mean_score = scores.mean()
    return mean_score
 
rf_bo = BayesianOptimization(_rf_evaluate, {'n_estimators': (500,800), 
                                            'max_features': (7,14), 
                                            'max_depth': (5,21), 
                                            'min_samples_leaf': (1,3), 
                                            'min_samples_split': (2,6), 
                                            }, allow_duplicate_points=True, random_state=90)

rf_bo.maximize(init_points=10, n_iter=100)

params_best =rf_bo.max["params"]
score_best = rf_bo.max["target"]

print('best params', params_best, '\n', 'best score', score_best)
reg_val = RandomForestRegressor(n_estimators= int(params_best["n_estimators"]), 
                                max_features=int(params_best["max_features"]),
                                max_depth=int(params_best["max_depth"]),
                                min_samples_leaf=int(params_best["min_samples_leaf"]),
                                min_samples_split=int(params_best["min_samples_split"]),
                                bootstrap=True,
                                random_state=90)

print('n_estimators:',int(params_best["n_estimators"]),
      'max_features',int(params_best["max_features"]),
      'max_depth',int(params_best["max_depth"]),
      'min_samples_leaf',int(params_best["min_samples_leaf"]),
      'min_samples_split',int(params_best["min_samples_split"]))

best_model=reg_val.fit(X_train,y_train)
print(best_model)
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

dump(best_model, 'model_RF.joblib')