import fasttext
import string
import json
import random
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.metrics import precision_recall_fscore_support
from gensim import corpora
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from FastText import FastText
import matplotlib.pyplot as plt
import seaborn as sns

class Individual():
    def __init__(self, epoch=1, lr=0, word_ngrams = 3, loss='ns', dim=300, ws=5,
                 precision=0, recall=0, fbeta=0, accuracy=0, confusion_matrix=None):
        self.epoch = epoch
        self.lr = lr
        self.word_ngrams = word_ngrams
        self.loss = loss
        self.dim = dim
        self.ws = ws
        self.precision = precision
        self.recall = recall
        self.fbeta = fbeta
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix

    def sort_individuals(self, individuals, param):
        k = lambda x: int(x['1'][param])
        individuals = sorted(individuals, key=k, reverse=True)
        return individuals


data = json.loads(open("parameters_all.json").read())
inds = []
for v in data.values():
    ind = Individual(v['ep'], v['lr'], v['ngrams'], v['loss'], v['dim'], v['ws'], v['precision'],
                     v['recall'], v['fbeta_score'], v['accuracy'], v['confusion_matrix'])
    inds.append(ind)
inds = sorted(inds, key=lambda x: x.confusion_matrix[0], reverse=True)
# inds = sorted(inds, key=lambda x: precision, reverse=True)
total_result = {}
count = 1
ep = 0
lr = 0
ngrams = 0
loss = {'ns':0, 'hs':0, 'softmax':0}
dim = 0
ws = 0
for i in inds:
    if i.precision > 0.5 and i.recall > 0.3:
        total_result[count] = {}
        total_result[count]['confusion_matrix'] = i.confusion_matrix
        total_result[count]['0precision'] = i.precision
        total_result[count]['recall'] = i.recall
        total_result[count]['fbeta_score'] = i.fbeta
        total_result[count]['accuracy'] = i.accuracy
        total_result[count]["ep"] = i.epoch
        total_result[count]["lr"] = i.lr
        total_result[count]["ngrams"] = i.word_ngrams
        total_result[count]["loss"] = i.loss
        total_result[count]["dim"] = i.dim
        total_result[count]["ws"] = i.ws
        ep += i.epoch
        lr += i.lr
        ngrams += i.word_ngrams
        loss[i.loss] = loss[i.loss]+1
        dim += i.dim
        ws += i.ws
        count += 1

f = open("parameters_all_sorted_tn.json", 'w+')
f.write(json.dumps(total_result, indent=4, sort_keys=True))
f.close()
print(ep/count)
print(lr/count)
print(ngrams/count)
print(loss)
print(dim/count)
print(ws/count)

color = sns.color_palette("hls", 8)
y1 = []
y2 = []
y3 = []
y4 = []
x = []
for i in range(1, count):
    x.append(i)
for i in x:
    y1.append(total_result[i]['0precision'])
    y2.append(total_result[i]['recall'])
    y3.append(total_result[i]['fbeta_score'])
    # y4.append(total_result[i]['accuracy'])

plt.figure(figsize=(20,12))
print(x)
print(len(x))
print(y1)
print(len(y1))
ax = sns.pointplot(x, y1, alpha=0.8, color=color[1], hue="precision")
ax = sns.pointplot(x, y2, alpha=0.8, color=color[2], hue="recall")
ax = sns.pointplot(x, y3, alpha=0.8, color=color[3], hue="f_score")
# sns.pointplot(x, y4, alpha=0.8, color=color[4])
# leg_handles = ax.get_legend_handles_labels()[0]
# ax.legend(leg_handles, ['precision', 'recall', 'f_score'], title='legend')

plt.ylabel('Evaluation', fontsize=12)
plt.xlabel('Parameters Combination Id', fontsize=12)
plt.title("FastText Performance Evaluation (recall sorted)", fontsize=15)
plt.xticks(rotation='vertical')
# plt.legend(['precision', 'recall', 'f_score'])
plt.show()
plt.savefig("FastText_tn.png")

# json1_file = open('parameters_all.json')
# json1_str = json1_file.read()
# json1_data = json.loads(json1_str)
# print(json1_data['1'])
# print(json1_data['2'])