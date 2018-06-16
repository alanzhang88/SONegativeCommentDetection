import argparse
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from IPython.display import HTML, display
import json

# Get all files 


y= []
parameters = []
with open("parameters_temp.json") as jsondata:
    data = json.load(jsondata)
    count = 2
    for d in data:
        if np.isnan(data[d]['precision']) or np.isnan(data[d]['recall']) or np.isnan(data[d]['f1']):
            count+=1
            continue
        y.append((count, data[d]['f1']))
        count+=1

color = sns.color_palette()

top100 = sorted(y, key=lambda x: x[1],reverse=True)[0:100]


dataplot1= {'precision':[],'recall':[],'f1':[]}

top100_id = []
for i in top100:
    top100_id.append(i[0])

with open("parameters_temp.json") as jsondata:
    data= json.load(jsondata)
    for id in top100_id:
        dataplot1['precision'].append(data[str(id)]['precision'])
        dataplot1['recall'].append(data[str(id)]['recall'])
        dataplot1['f1'].append(data[str(id)]['f1'])
x = list(range(100))



plt.figure(figsize=(8,6))
sns.pointplot(x, dataplot1['precision'], alpha=0.1, color=color[1])
sns.pointplot(x, dataplot1['recall'], alpha=0.2, color=color[2])
sns.pointplot(x, dataplot1['f1'], alpha=0.3, color=color[3])

color_patch1 = mpatches.Patch(color=color[1], label="precision")
color_patch2 = mpatches.Patch(color=color[2], label="recall")
color_patch3 = mpatches.Patch(color=color[3], label="f1")
plt.legend(handles=[color_patch1,color_patch2,color_patch3])
plt.ylabel('Evaluation', fontsize=12)
plt.xlabel('Parameters combination', fontsize=12)
plt.title("CNN", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()