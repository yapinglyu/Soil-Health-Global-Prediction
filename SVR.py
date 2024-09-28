import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from joblib import dump, load


file_path = "datapreprocessing.xlsx"
data = pd.read_excel(file_path, sheet_name='pH') # change any sheet name to get the data
X=data.iloc[:,0:19].values
y=data.iloc[:,-1].values
print(X[:1])
print(y[:1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)


def _svr_evaluate(C, gamma): 
    reg = SVR (C=C,
              gamma=gamma,
              kernel='rbf')
    kf = KFold(n_splits=5)
    cv_result = cross_val_score(reg, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    return np.mean(cv_result)

svr_bo = BayesianOptimization(_svr_evaluate, {'C': (1,800), 
                                              'gamma': (0,0.50), 
                                              }, random_state=40)
svr_bo.maximize(init_points=10, n_iter=200) 

params_best =svr_bo.max["params"]
score_best = svr_bo.max["target"]

print('best params', params_best, '\n', 'best score', score_best)

reg_val = SVR(C=params_best["C"],
              gamma=params_best["gamma"],
              kernel='rbf')

print('C:',params_best["C"],
    'gamma=',params_best["gamma"],)

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


dump(best_model, 'model_SVR.joblib')

