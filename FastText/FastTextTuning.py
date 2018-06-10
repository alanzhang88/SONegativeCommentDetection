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


def create_classifier(train_file, model_name, epoch, lr, ngrams, loss, dim, ws):
    return fasttext.supervised(train_file,
                               model_name,
                               epoch=epoch,
                               dim=dim,
                               word_ngrams=ngrams,
                               lr=lr,
                               ws=ws,
                               min_count=5,
                               bucket=2000000,
                               loss=loss)
count = 1
total_result = {}
epochs = [1, 5, 10]
lr = [0.1, 0.3, 0.5]
word_ngrams = [1, 2, 3]
loss = ['ns', 'hs', 'softmax']
dim = [100, 200, 300]
ws = [1, 5, 10]

# total_result = {}
# epochs = [1]
# lr = [0.1]
# word_ngrams = [1]
# loss = ['ns', 'hs', 'softmax']
# dim = [100]
# ws = [1]

file = '../Models/LSTM/labeled_document_seconditer.json'
train_file = 'training_seconditer.txt'
test_file = 'testing_seconditer.txt'
model = FastText()
(postProcessedTrainPhrases, postProcessedTestPhrases, trainLabels, testLabels) = model.preprocessData(file)
testSentences = model.extractText(postProcessedTestPhrases)

epoch_choice= 5
# lr_choice = 0.3
dim_choice = 200
ngrams_choice = 3
loss_choice = 'ns'
ws_choice = 5
for i in range(1,4):
    for lr_choice in lr:
        classifier = fasttext.supervised('training_seconditer.txt', 'model_',epoch=epoch_choice, lr=lr_choice, dim=dim_choice,
                                                 word_ngrams=ngrams_choice,loss=loss_choice, ws=ws_choice, min_count=5,bucket=2000000)
        print(count)
        res = classifier.predict(testSentences)
        predicted = []
        for i in res:
            predicted.append(int(i[0]))
        # print(predicted)
        total_result[count] = {}
        tn, fp, fn, tp = confusion_matrix(testLabels, predicted).ravel()
        # print(tn, fp, fn, tp)
        total_result[count]['confusion_matrix'] = []
        total_result[count]['confusion_matrix'].append(int(tn))
        total_result[count]['confusion_matrix'].append(int(fp))
        total_result[count]['confusion_matrix'].append(int(fn))
        total_result[count]['confusion_matrix'].append(int(tp))
        # print(total_result[count]['confusion_matrix'])
        report = precision_recall_fscore_support(testLabels, predicted)
        print(report)
        total_result[count]['precision'] = report[0][0]
        total_result[count]['recall'] = report[1][0]
        total_result[count]['fbeta_score'] = report[2][0]
        # print(report)
        # print(report.fbeta_score)
        total_result[count]['accuracy'] = classifier.test(test_file).precision
        total_result[count]["ep"] = epoch_choice
        total_result[count]["lr"] = lr_choice
        total_result[count]["ngrams"] = ngrams_choice
        total_result[count]["loss"] = loss_choice
        total_result[count]["dim"] = dim_choice
        total_result[count]["ws"] = ws_choice
        # total_result[count]['model'] = [epoch_choice, lr_choice,ngrams_choice,loss_choice, dim_choice,ws_choice]
        f = open("parameters_temp.json", 'w')
        f.write(json.dumps(total_result, indent=4, sort_keys=True))
        f.close()
        count += 1

f = open("parameters_lr.json", 'w+')
f.write(json.dumps(total_result, indent=4, sort_keys=True))
f.close()

# with open("parameters_temp.json", mode='w', encoding='utf-8') as f:
# for epoch_choice in epochs:
#     for lr_choice in lr:
#         for ngrams_choice in word_ngrams:
#             for loss_choice in loss:
#                 for dim_choice in dim:
#                     for ws_choice in ws:
#                         model_name = "model_"
#                         classifier = create_classifier(train_file, model_name, epoch_choice, lr_choice,
#                                                        ngrams_choice, loss_choice, dim_choice, ws_choice)
#                         testSentences = FastText().extractText(postProcessedTestPhrases)
#                         res = classifier.predict(testSentences)
#                         predicted = []
#                         for i in res:
#                             predicted.append(int(i[0]))
#                         # print(predicted)
#                         total_result[count] = {}
#                         tn, fp, fn, tp = confusion_matrix(testLabels, predicted).ravel()
#                         # print(tn, fp, fn, tp)
#                         total_result[count]['confusion_matrix'] = []
#                         total_result[count]['confusion_matrix'].append(int(tn))
#                         total_result[count]['confusion_matrix'].append(int(fp))
#                         total_result[count]['confusion_matrix'].append(int(fn))
#                         total_result[count]['confusion_matrix'].append(int(tp))
#                         # print(total_result[count]['confusion_matrix'])
#                         report = precision_recall_fscore_support(testLabels, predicted)
#                         total_result[count]['precision'] = report[0][0]
#                         total_result[count]['recall'] = report[1][0]
#                         total_result[count]['fbeta_score'] = report[2][0]
#                         # print(report)
#                         # print(report.fbeta_score)
#                         total_result[count]['accuracy'] = classifier.test(test_file).precision
#                         total_result[count]["ep"] = epoch_choice
#                         total_result[count]["lr"] = lr_choice
#                         total_result[count]["ngrams"] = ngrams_choice
#                         total_result[count]["loss"] = loss_choice
#                         total_result[count]["dim"] = dim_choice
#                         total_result[count]["ws"] = ws_choice
#                         # total_result[count]['model'] = [epoch_choice, lr_choice,ngrams_choice,loss_choice, dim_choice,ws_choice]
#                         f = open("parameters_temp.json", 'w')
#                         f.write(json.dumps(total_result, indent=4, sort_keys=True))
#                         f.close()
#                         count += 1

# f = open("parameters_all.json", 'w+')
# f.write(json.dumps(total_result, indent=4, sort_keys=True))
# f.close()

color = sns.color_palette("hls", 8)
y1 = []
y2 = []
y3 = []
y4 = []
x = []
for i in range(1, count):
    x.append(i)
for i in x:
    y1.append(total_result[i]['precision'])
    y2.append(total_result[i]['recall'])
    y3.append(total_result[i]['fbeta_score'])
    # y4.append(total_result[i]['accuracy'])

plt.figure(figsize=(20,12))
print(x)
print(len(x))
print(y1)
print(len(y1))
sns.pointplot(x, y1, alpha=0.8, color=color[1])
sns.pointplot(x, y2, alpha=0.8, color=color[2])
sns.pointplot(x, y3, alpha=0.8, color=color[3])
# sns.pointplot(x, y4, alpha=0.8, color=color[4])

plt.ylabel('Evaluation', fontsize=12)
plt.xlabel('Parameters Combination Id', fontsize=12)
plt.title("FastText Performance Evaluation", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
plt.savefig("FastText.png")

# json1_file = open('parameters_all.json')
# json1_str = json1_file.read()
# json1_data = json.loads(json1_str)
# print(json1_data['1'])
# print(json1_data['2'])
# print(json1_data['3'])