import argparse
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from IPython.display import HTML, display


# Get all files 
files = glob.glob('*.txt')

dataplot1= {'precision':[],'recall':[],'f1-score':[]}
color = sns.color_palette()

combination = {"fields":[], "f1_score":[]}


for f in files:
    # get combination
    combination = re.findall(r'\d+', f)
    f = open(f, 'r')
    line = f.readline()
    line_arr = line.split("- ")
    precision = line_arr[8].split(":")[1]
    recall = line_arr[9].split(":")[1]
    f1_score = line_arr[10].split(":")[1]
    dataplot1["precision"].append(precision)
    dataplot1["recall"].append(recall)
    dataplot1["f1_score"].append(f1_score)
    combination["fields"].append(combination)
    combination["f1_score"].append(f1_score)



top_100 = sorted(dataplot1['f1_score'],reverse=True)[0:100]

x = list(range(100))

plt.figure(figsize=(20,12))
sns.pointplot(x, dataplot1['precision'], alpha=0.1, color=color[1])
sns.pointplot(x, dataplot1['recall'], alpha=0.2, color=color[2])
sns.pointplot(x, dataplot1['f1-score'], alpha=0.3, color=color[3])

color_patch1 = mpatches.Patch(color=color[1], label="precision")
color_patch2 = mpatches.Patch(color=color[2], label="recall")
color_patch3 = mpatches.Patch(color=color[3], label="fbeta_score")
plt.legend(handles=[color_patch1,color_patch2,color_patch3])
plt.ylabel('Evaluation', fontsize=12)
plt.xlabel('Parameters combination', fontsize=12)
plt.title("Single Stack LSTM", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
    






