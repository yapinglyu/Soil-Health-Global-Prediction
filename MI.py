from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_path = "datapreprocessing.xlsx"
data = pd.read_excel(file_path, sheet_name='pH') # change any sheet name to get the data
X=data.iloc[:,0:15]
y=data.iloc[:,-1]
features=X.columns
n_runs = 10


results = []
mi_matrix = pd.DataFrame(np.zeros((len(features), len(features))), index=features, columns=features)
for _ in range(n_runs):
    mi = mutual_info_regression(X, y)
    results.append(mi)

mi_mean = np.mean(results, axis=0)
mi_std = np.std(results, axis=0)

df = pd.DataFrame({
    'Mutual Information 1': mi_mean,
    'Standard Deviation 1': mi_std
}, index=X.columns)

for i in range(len(features)):
    for j in range(i + 1, len(features)):  
        mi = mutual_info_regression(data[[features[i]]], data[features[j]])[0]
        mi_matrix.iloc[i, j] = mi

df1 = pd.DataFrame((mi_matrix))
with pd.ExcelWriter("mi_matrix.xlsx") as writer:
    df.to_excel(writer, sheet_name='MI_with_label')
    df1.to_excel(writer, sheet_name='MI_with_feature')