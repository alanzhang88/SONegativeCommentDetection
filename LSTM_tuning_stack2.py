
# coding: utf-8

# In[1]:


import numpy as np
import string
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Bidirectional, GlobalMaxPool1D
from keras.layers.core import SpatialDropout1D
from sklearn.model_selection import StratifiedKFold
from keras.datasets import imdb
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import model_from_json

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

from gensim import corpora
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# def parsePhrases(stopWords, engStemmer, phrases):
#     print "parse the phrases with stopwords and stemmer"
#     processedPhrases = []
#     for phrase in phrases:
#         tokens = word_tokenize(phrase)
#         parsedWords = []
#         for t in tokens:
#             if t not in stopWords:
#                 parsedWords.append(engStemmer.stem(t))
#         processedPhrases.append(parsedWords)
#     return processedPhrases
postProcessedTrainPhrases = []
postProcessedTestPhrases = []

def preprocessData():
    print("Loading and preprocessing data...")
    # load training and testing data
    with open('labeled_document_firstiter.json') as json_data:
        allTrainData = json.load(json_data)
    
    with open('labeled_document_seconditer.json') as json_data:
        allTrainData2 = json.load(json_data)

    
    trainPhrases, testPhrases, trainLabel,testLabel = train_test_split(allTrainData['Comment'] + allTrainData2['Comment'], allTrainData['CommentLabel']+allTrainData2['CommentLabel'], test_size=0.2, random_state=42)
    
#     print(testPhrases[0:100])
    punctuation = list(string.punctuation)
    stopWords = stopwords.words('english') + punctuation 

    engStemmer = SnowballStemmer('english')
    for phrase in trainPhrases:
        if not isinstance(phrase, str):
            continue
        tokens = word_tokenize(phrase)
        parsedWords = []
        for t in tokens:
            if t not in stopWords:
                parsedWords.append(engStemmer.stem(t))
        postProcessedTrainPhrases.append(parsedWords)

    for phrase in testPhrases:
        if not isinstance(phrase, str):
            continue
        tokens = word_tokenize(phrase)
        parsedWords = []
        for t in tokens:
            if t not in stopWords:
                parsedWords.append(engStemmer.stem(t))
        postProcessedTestPhrases.append(parsedWords)
    return (trainLabel,testLabel)


def convertPhrasesToIDs(phrases):
    print ("converting the phrases to id to be processed")
    wordIDs = []
    wordIDLens = []
    for phrase in phrases:
        ids = []
        for word in phrase:
            ids.append(toIDMap.token2id[word])
        wordIDs.append(ids)
        wordIDLens.append(len(ids))
    return ( wordIDs, wordIDLens )

def findSequenceLen(wordListLen):
    print( "calculate the norm sequence length")
    wordLenMean = np.mean(wordListLen)
    wordLenStd = np.std(wordListLen)
    return np.round(wordLenMean + 3 * wordLenStd).astype(int)



# In[2]:


(trainSenti, testSenti) = preprocessData()

# process training data and testing data

# print(len(postProcessedTrainPhrases), len(trainSenti))
toIDMap = corpora.Dictionary(np.concatenate((postProcessedTrainPhrases, postProcessedTestPhrases), axis=0))
allPhraseSize = len(toIDMap.keys())

(trainWordIDs, trainWordIDLens) = convertPhrasesToIDs(postProcessedTrainPhrases)
(testWordIDs, testWordIDLens) = convertPhrasesToIDs(postProcessedTestPhrases)

sequenceLen = findSequenceLen(trainWordIDLens + testWordIDLens)

print( "pad sequence")
trainingData = sequence.pad_sequences(np.array(trainWordIDs), maxlen=sequenceLen)
testingData = sequence.pad_sequences(np.array(testWordIDs), maxlen=sequenceLen)
print(trainingData.shape)

print ("categorize the labels")
#print len(np.unique(trainSenti))
trainingDataLabel = np_utils.to_categorical(trainSenti, len(np.unique(trainSenti)))

# print(trainingDataLabel.shape)



# In[6]:


from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.metrics import precision_recall_fscore_support

# def create_model(hidden_size, activation, optimizer, dropout_rate):
#     # default values
#     # activation='tanh' # or linear
#     # dropout_rate=0.0 # or 0.2
#     init_mode='uniform'
#     weight_constraint=0 # or  4
#     # optimizer='adam' # or SGD
#     lr = 0.01
#     momemntum=0
#     # hidden_size = 128
#     # create model
#     model = Sequential()
#     model.add(Embedding(allPhraseSize, embedding_size))
#     model.add(SpatialDropout1D(dropout_rate))
# #     model.add(Dense(8, 
# #                     input_dim=input_dim, kernel_initializer=init_mode, 
# #                     activation=activation,
# #                     kernel_constraint=maxnorm(weight_constraint)))
# #     model.add(Dropout(dropout_rate)) 
#     model.add(Bidirectional(LSTM(hidden_size, activation=activation)))
#     model.add(Dense(2, kernel_initializer=init_mode))
#     model.add(Activation(activation))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', 
#                   optimizer=optimizer, 
#                   metrics=['accuracy'])
#     return model

# model = KerasClassifier(build_fn=create_model, batch_size=100, epochs=10) 
# epochs = [5, 10, 50, 100, 500]
# optimizer = ['sgd', 'RMSprop', 'adam']
# activation = ['tanh','softmax','relu','sigmoid']
# hid_size = [64, 128, 256]
# dropoutrate = [0.0, 0.05, 0.1, 0.25, 0.5]
embedding_size = 128
# parameters = {'optimizer':('sgd', 'RMSprop', 'adam'), 'activation':[1, 10]}
activation =  ['sigmoid', 'hard_sigmoid','softmax'] # softmax, softplus, softsign 
# hidden_size = [ 128, 256]
# momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
# learn_rate = [0.001, 0.01, 0.1, 0.2]
dropout_rate = [0.1,  0.5]
# weight_constraint=[1, 2, 3, 4, 5]
# neurons = [1, 5, 10, 15, 20, 25, 30]
# init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
optimizer = [ 'SGD', 'RMSprop', 'Adam']
epochs = [10, 100] 
batch_size = [256]
HIDDEN_SIZE = 128
# param_grid = dict(epochs=epochs, batch_size=batch_size, activation = activation, dropout_rate = dropout_rate, optimizer = optimizer, hidden_size = hidden_size)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(trainingData,trainingDataLabel) 
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
count = 1
total_result = {}
testingDataLabel = np_utils.to_categorical(testSenti, len(np.unique(testSenti)))

# with open("parameters_temp.json", mode='w', encoding='utf-8') as f:
for epoch_choice in epochs:
    for batch_choice in batch_size:
        for activation_choice in activation:
            for dropoutrate in dropout_rate:
                for optimizer_choice in optimizer:  
                    # for HIDDEN_SIZE in hidden_size:          
                    model = Sequential()
                    model.add(Embedding(allPhraseSize, embedding_size))
                    model.add(SpatialDropout1D(dropoutrate))
                    model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True)))
                    model.add(Bidirectional(LSTM(HIDDEN_SIZE)))
                    #model.add(Bidirectional(LSTM(128)))
                    #model.add(Flatten())
                    model.add(Dense(len(np.unique(trainSenti))))
                    model.add(Activation(activation_choice))
                    # model.add(CRF(2, sparse_target=True))

                    model.compile(loss='categorical_crossentropy', optimizer=optimizer_choice, metrics=['accuracy'])

                    model.fit(trainingData,trainingDataLabel , epochs=epoch_choice, batch_size=batch_choice, verbose=1)

                    res = model.predict(testingData)
                    res = [(np.array(l)/sum(l)).tolist() for l in res]
                    # print(predicted)
                    predicted = []
                    # negcount = 0
                    # poscount = 0
                    for i in res:
                        if i[0] > i[1]:
                            # negcount +=1
                            predicted.append(0)
                        else:
                            # poscount +=1
                            predicted.append(1)
                    # print(predicted)
                    total_result[count] = {}

                    tn, fp, fn, tp = confusion_matrix(testSenti, predicted).ravel()
                    print(tn, fp, fn, tp)
                    total_result[count]['confusion_matrix'] = []
                    total_result[count]['confusion_matrix'].append(int(tn))
                    total_result[count]['confusion_matrix'].append(int(fp))
                    total_result[count]['confusion_matrix'].append(int(fn))
                    total_result[count]['confusion_matrix'].append(int(tp))
                    print(total_result[count]['confusion_matrix'])
                    report = precision_recall_fscore_support(testSenti, predicted)
                    total_result[count]['precision'] = report[0][0]
                    total_result[count]['recall'] = report[1][0]
                    total_result[count]['fbeta_score'] = report[2][0]
                    print(report)
                    # print(report.fbeta_score)
                    scores = model.evaluate(testingData, testingDataLabel, verbose=0)
                    total_result[count]['accuracy'] = scores[1] * 100
                    total_result[count]["ep"] = epoch_choice
                    total_result[count]["batch"] = batch_choice
                    total_result[count]["act"] = activation_choice
                    total_result[count]["drop"] = dropoutrate
                    total_result[count]["op"] = optimizer_choice
                    # total_result[count]["hid"] = HIDDEN_SIZE
                    # total_result[count]['model'] = [epoch_choice, batch_choice,activation_choice,dropoutrate, optimizer_choice,HIDDEN_SIZE]
                    f = open("parameters_temp_stack2.json", 'w')
                    f.write(json.dumps(total_result, indent=4, sort_keys=True))
                    f.close()
                    count += 1

f = open("parameters_all_stack2.json", 'w+')
f.write(json.dumps(total_result, indent=4, sort_keys=True))
f.close()
# In[83]:
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
y1 = []
y2 = []
y3 = []
y4 = []
x = range(1,count)
for i in x:
    y1.append(total_result[i]['precision'])
    y2.append(total_result[i]['recall'])
    y3.append(total_result[i]['fbeta_score'])
    y4.append(total_result[i]['accuracy'])

plt.figure(figsize=(20,12))
sns.pointplot(x, y1, alpha=0.8, color=color[1])
sns.pointplot(x, y2, alpha=0.8, color=color[2])
sns.pointplot(x, y3, alpha=0.8, color=color[3])
sns.pointplot(x, y4, alpha=0.8, color=color[4])

plt.ylabel('Evaluation', fontsize=12)
plt.xlabel('Parameters combination', fontsize=12)
plt.title("Single Stack LSTM", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()
plt.savefig("LSTM1.png")

# from sklearn.metrics import precision_recall_fscore_support
# res = model.predict(testingData)
# res = [(np.array(l)/sum(l)).tolist() for l in res]
# # print(predicted)
# predicted = []
# negcount = 0
# poscount = 0
# for i in res:
#     if i[0] > i[1]:
#         negcount +=1
#         predicted.append(0)
#     else:
#         poscount +=1
#         predicted.append(1)

# print("negative: ", negcount)
# print("positive: ", poscount)

# matrix = confusion_matrix(testSenti, predicted)
# print(matrix)
# report = precision_recall_fscore_support(testSenti, predicted)
# print("precision: ", report[0][0])
# print("recall: ", report[1][0])
# print("fbeta_score: ",report[2][0] )
# print(report.recall)
# # print(report.fbeta_score)
# scores = model.evaluate(testingData, testingDataLabel, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


# # In[29]:


# model_json = model.to_json()
# with open("LSTM.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("LSTM.h5")
# print("Saved model to disk")


# # In[37]:


# json_file = open('LSTM.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("LSTM.h5")
# print("Loaded model from disk")


# 1. Number of hidden layers
# 2. Number of hidden units per layer (usually same number in each layer)
# 3. Learning rate of the optimizer
# 4. Dropout rate (in RNNs dropout is perhaps better applied to feed forward connections only)
# 5. Number of iterations

# 1 lstm
# negative:  310
# positive:  1936
# [[ 186  159]
#  [ 124 1777]]
# precision:  0.6
# recall:  0.5391304347826087
# fbeta_score:  0.5679389312977099
# acc: 87.40%
# 
# negative:  382
# positive:  1864
# [[ 211  134]
#  [ 171 1730]]
# precision:  0.5523560209424084
# recall:  0.6115942028985507
# fbeta_score:  0.5804676753782669
# acc: 86.42%

# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
