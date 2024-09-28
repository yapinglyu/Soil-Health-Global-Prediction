import numpy as np  # 用来做数学运算
import pandas as pd  # 用来处理数据表
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split #划分测试集和训练集
from sklearn.model_selection import ShuffleSplit #分折交叉验证
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import r2_score,make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from math import sqrt
from sklearn.inspection import PartialDependenceDisplay
from matplotlib import pyplot as plt
from pdpbox import pdp, info_plots


# 读取文件原始数据
file_path = "datapreprocessing.xlsx"
data = pd.read_excel(file_path, sheet_name='pH') # change any sheet name to get the data
X=data.iloc[:,0:19]
y=data.iloc[:,-1]

#take the pH prediction model as example 
model = xgb.XGBRegressor(n_estimators=572, 
                         max_depth=6, 
                         learning_rate=0.08, 
                         subsample=0.8, 
                         gamma=0,
                         colsample_bytree=0.41212210400773,
                         min_child_weight=0.136446684379273, 
                         eta=0.19115830548605,
                         alpha=0.110812222346563, 
                         random_state=40)
model.fit(X, y)

pdp_interact = pdp.pdp_interact(
    model=model, 
    dataset=X, 
    model_features=X.columns, 
    features=['SSA', 'SC'],
    num_grid_points=[600, 600]
)
fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=pdp_interact, 
    feature_names=['SSA', 'SC'], 
    plot_type='contour', 
    x_quantile=True, 
    plot_pdp=True
)
plt.show()

pdp_values = pdp_interact.pdp

df = pd.DataFrame(pdp_values)
df.to_excel("D:\论文集\总结\论文6-UoB\MLcode\pdp_values.xlsx")

