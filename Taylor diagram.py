import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patches as patches
import numpy as np
import pandas  as pd
import skill_metrics as sm
 

data_pH = pd.read_excel("taylor diagram data.xlsx", sheet_name='pH') # change any sheet name to get the data
taylor_stats1 = sm.taylor_statistics(data_pH.XGBoost,data_pH.ref,'data')
taylor_stats2 = sm.taylor_statistics(data_pH.RF,data_pH.ref,'data')
taylor_stats3 = sm.taylor_statistics(data_pH.SVR,data_pH.ref,'data')
taylor_stats4 = sm.taylor_statistics(data_pH.GBDT,data_pH.ref,'data')
sdev = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1], 
                 taylor_stats2['sdev'][1], taylor_stats3['sdev'][1], taylor_stats4['sdev'][1]])
crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], 
                  taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1], taylor_stats4['crmsd'][1]])
ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1], 
                  taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1], taylor_stats4['ccoef'][1]])
 

rcParams["figure.figsize"] = [7.1, 7]
rcParams["figure.facecolor"] = "white"
rcParams["figure.edgecolor"] = "white"
rcParams["figure.dpi"] = 80
rcParams['lines.linewidth'] = 1 # 
rcParams["font.family"] = "Times New Roman"
rcParams.update({'font.size': 20}) # 
plt.close('all')

print("sedv:", sdev)
print("crmsd:", crmsd)
print("ccoef:", ccoef)
text_font = {'size':'15','weight':'bold','color':'black'}

sm.taylor_diagram(sdev,crmsd,ccoef, 
                  axismax=4,
                  markerLabel=['M1','XGBoost','RF','SVR','GBDT']
                  )
ref_std = sdev[0]
circle = patches.Circle((0, 0), ref_std, fill=False, linestyle='dashed', linewidth=0.5, edgecolor='red')
plt.gca().add_patch(circle)
plt.title("pH",fontdict=text_font,pad=40, fontsize='medium')
plt.savefig("pH.tiff", dpi=600)
plt.show()

