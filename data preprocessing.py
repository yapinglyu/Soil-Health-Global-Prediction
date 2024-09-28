from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from openpyxl import load_workbook

data_pH = pd.read_excel('preprocessing.xlsx', sheet_name='pH') # change any sheet name to get the data
X_pH=data_pH.iloc[:,8:19].values
y_pH=data_pH.iloc[:,-1].values
print(X_pH[:3])
imputer = SimpleImputer(strategy='most_frequent')
X_pH[:, 0] = imputer.fit_transform(X_pH[:, 0].reshape(-1, 1)).ravel()
X_pH[:, 3] = imputer.fit_transform(X_pH[:, 3].reshape(-1, 1)).ravel()
X_pH[:, 9] = imputer.fit_transform(X_pH[:, 9].reshape(-1, 1)).ravel()
print(X_pH[:3])
df_pH = pd.DataFrame(X_pH)
df_pH.to_excel("fill missing data pH.xlsx", sheet_name='pH', index=False)


encoder = OneHotEncoder()
ct = ColumnTransformer(
    [("encoder", encoder, [0, 14])], 
    remainder='passthrough'  
)
X2 = ct.fit_transform(X_pH)

feature_names = ct.named_transformers_['encoder'].get_feature_names_out(['column1', 'column15'])
print(feature_names)
print(X2[:1])

data2_pH = pd.read_excel('ill missing data pH.xlsx', sheet_name='pH')
X2_pH=data2_pH.iloc[:,0:11].values
print(X2_pH[:3])
skewness = data_pH.skew()
print("skewness：\n", skewness)
skewness.plot(kind='bar', title='Feature Skewness')

#standardization
data3_pH = pd.read_excel('fill missing data pH.xlsx', sheet_name='pH')
X3_pH=data3_pH.iloc[:,0:11].values
scaler = StandardScaler()
X3_pH[:, 0:11] = scaler.fit_transform(X3_pH[:, 0:11])


print(X3_pH[:1])
df = pd.DataFrame(X3_pH)
file_path = "datapreprocessing pH.xlsx"
df.to_excel(file_path, sheet_name='pH', index=False)

