import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import cross_val_score, KFold
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump, load


file_path = "datapreprocessing.xlsx"
data = pd.read_excel(file_path, sheet_name='pH') # change any sheet name to get the data
X=data.iloc[:,0:19].values
y=data.iloc[:,-1].values
print(X[:1])
print(y[:1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

def gbdt_eval(n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf):
    estimator = GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=2
    )
    cv_score = cross_val_score(estimator, X_train, y_train, scoring='neg_mean_squared_error', cv=KFold(n_splits=5))
    return np.mean(cv_score)

gbdt_opt = BayesianOptimization(
    gbdt_eval, 
    {'n_estimators': (100, 250),
     'learning_rate': (0.01, 0.05),
     'max_depth': (3, 7),
     'min_samples_split': (2, 7),
     'min_samples_leaf': (1, 7),
    },
    random_state=123
)

gbdt_opt.maximize(n_iter=10, init_points=50)

params = gbdt_opt.max['params']
params['max_depth'] = int(params['max_depth'])
params['min_samples_split'] = int(params['min_samples_split'])
params['min_samples_leaf'] = int(params['min_samples_leaf'])
params['n_estimators'] = int(params['n_estimators'])

best_gbdt = GradientBoostingRegressor(**params)
best_model=best_gbdt.fit(X_train,y_train)

print(params)


y_train_pred = best_gbdt.predict(X_train)
y_test_pred = best_gbdt.predict(X_test)

print('trainMSE:', mean_squared_error(y_train, y_train_pred))
print('trainRMSE:', np.sqrt(mean_squared_error(y_train, y_train_pred)))
print('trainMAE:', mean_absolute_error(y_train, y_train_pred))
print('testMSE:', mean_squared_error(y_test, y_test_pred))
print('testRMSE:', np.sqrt(mean_squared_error(y_test, y_test_pred)))
print('trainMAE:', mean_absolute_error(y_test, y_test_pred))
print('train-r2', r2_score(y_train, y_train_pred))
print('test-r2', r2_score(y_test, y_test_pred))
print(best_model.feature_importances_)


dump(best_model, 'model_GBDT.joblib')
