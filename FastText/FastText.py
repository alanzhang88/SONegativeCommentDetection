import fasttext
import string
import json
import random
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
# from imblearn.over_sampling import SMOTE

# # Skipgram model
# model = fasttext.skipgram('data.txt', 'model')
# print(model.words) # list of words in dictionary
#
# # CBOW model
# model = fasttext.cbow('data.txt', 'model')
# print(model.words) # list of words in dictionary

class FastText:
    postProcessedTrainPhrases = []
    postProcessedTestPhrases = []
    trainSentences = []

    def __init__(self):
        print("init")

    def preprocessData(self, file):
        postProcessedTrainPhrases = []
        postProcessedTestPhrases = []
        trainSentences = []
        print("Loading and preprocessing data...\n")
        # load training and testing data
        with open('../Models/LSTM/labeled_document_firstiter.json') as json_data:
            allTrainData = json.load(json_data)

        with open('../Models/LSTM/labeled_document_seconditer.json') as json_data:
            allTrainData2 = json.load(json_data)

        trainPhrases, testPhrases, trainLabel, testLabel = train_test_split(
            allTrainData['Comment'] + allTrainData2['Comment'],
            allTrainData['CommentLabel'] + allTrainData2['CommentLabel'], test_size=0.2, random_state=42)
        # with open(file) as json_data:
        #     allTrainData = json.load(json_data)
        #
        # trainPhrases, testPhrases, trainLabel, testLabel = train_test_split(allTrainData['Comment'],
        #                                                                     allTrainData['CommentLabel'], test_size=0.2,
        #                                                                     random_state=42)

        #     print(testPhrases[0:100])
        punctuation = list(string.punctuation)
        stopWords = stopwords.words('english') + punctuation

        engStemmer = SnowballStemmer('english')
        # postProcessedTrainPhrases = []
        #     for phrase in trainPhrases:
        #         uni_doc = unicode(phrase, errors='replace')
        #         tokens = word_tokenize(uni_doc)
        #         filtered = [word for word in tokens if word not in stop_words]
        #         try:
        #             stemmed = [stemmer.stem(word) for word in filtered]
        #         except UnicodeDecodeError:
        #             print(word)
        #         postProcessedTrainPhrases.append(parsedWords)

        #     for phrase in testPhrases:
        #         uni_doc = unicode(phrase, errors='replace')
        #         tokens = word_tokenize(uni_doc)
        #         filtered = [word for word in tokens if word not in stop_words]
        #         try:
        #             stemmed = [stemmer.stem(word) for word in filtered]
        #         except UnicodeDecodeError:
        #             print(word)
        #         postProcessedTestPhrases.append(parsedWords)
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
            trainSentences.append(phrase)
            tokens = word_tokenize(phrase)
            parsedWords = []
            for t in tokens:
                if t not in stopWords:
                    parsedWords.append(engStemmer.stem(t))
            postProcessedTestPhrases.append(parsedWords)
        return (postProcessedTrainPhrases, postProcessedTestPhrases, trainLabel, testLabel)

    def outputPhrasesToFile(self, filename, phrases, labels):
        f = open(filename+".txt", "w+")
        for i in range(len(phrases)):
            sentence = ""
            for j in range(len(phrases[i])):
                sentence += " " + phrases[i][j]
            f.write("__label__" + str(labels[i]) + sentence + "\r\n")
        f.close()

    def outputSentencesToFile(self, filename, sentences, labels):
        f = open(filename+".txt", "w+")
        for i in range(len(sentences)):
            f.write("__label__" + str(labels[i]) + " " + sentences[i] + "\r\n")
        f.close()

    def extractText(self, phrases):
        sentences = []
        for i in range(len(phrases)):
            sentence = ""
            for j in range(len(phrases[i])):
                sentence += " " + phrases[i][j]
            sentences.append(sentence)
        return sentences


    def downsampling(self, phrases, labels, minority):
        sentences = self.extractText(phrases)
        majority = []
        sentencesDS = []  # downsampleing (sentences with label 1 : 0 = 1 : 1)
        labelsDS = []
        for i in range(len(labels) - 1 ):
            if labels[i] == minority:
                sentencesDS.append(sentences[i])
                labelsDS.append(0)
            else:
                majority.append(sentences[i])
        minorityLen = len(sentencesDS)
        majority = random.sample(majority, minorityLen)
        for m in majority:
            sentencesDS.append(m)
            labelsDS.append(1)
        return sentencesDS, labelsDS

    def load_model(self):
        model = fasttext.load_model('best_model.bin')
        return model

    def predict(self, texts):
        model = fasttext.load_model('best_model.bin')
        labels = model.predict_proba(texts)
        results = []
        for label in labels:
            tmp = []
            if (int(label[0][0][-1]) == 0):
                tmp.append(label[0][1])
                tmp.append(1.0-label[0][1])
            else:
                tmp.append(1.0 - label[0][1])
                tmp.append(label[0][1])
            results.append(tmp)
        print(results)

    def classify(self, file, isDS):
        # preprocess data
        (postProcessedTrainPhrases, postProcessedTestPhrases, trainLabels, testLabels) = self.preprocessData(file)

        # create training and testing file
        if (isDS):
            trainSentencesDS, trainLabelsDS = self.downsampling(postProcessedTrainPhrases, trainLabels, 0)
            self.outputSentencesToFile("training_all", trainSentencesDS, trainLabelsDS)
            testSentencesDS, testLabelsDS = self.downsampling(postProcessedTestPhrases, testLabels, 0)
            self.outputSentencesToFile("testing_all", testSentencesDS, testLabelsDS)
        else:
            self.outputPhrasesToFile("training_all", postProcessedTrainPhrases, trainLabels)
            self.outputPhrasesToFile("testing_all", postProcessedTestPhrases, testLabels)
            testSentences = self.extractText(postProcessedTestPhrases)

        # train the fasttext model
        print('Buidling the model...\n')
        # sm = SMOTE(random_state=12, ratio = 1.0)
        # trainingData, trainLabels = sm.fit_sample(np.array(trainSentences).reshape(len(trainSentences), 1), trainLabels)
        if (isDS):
            #   downsampling inbalanced data
            classifier = fasttext.supervised('trainingDS_all.txt', 'best_model')
            result = classifier.test('testingDS_all.txt')
        else:
            #   without downsampling inbalanced data
            classifier = fasttext.supervised('training_all.txt', 'best_model',epoch=5, lr=0.1, dim=100,
                                             word_ngrams=1,loss='hs', ws=5, min_count=5,bucket=2000000)
            # classifier = fasttext.supervised('training_seconditer.txt', 'model_seconditer', epoch=5)
            result = classifier.test('testing_all.txt')

        # classify testing data
        if (isDS):
            #   downsampling inbalanced data
            labels = classifier.predict(testSentencesDS)
        else:
            #   without downsampling inbalanced data
            labels = classifier.predict(testSentences)

        print('Negative comments found:')
        for i in range(len(labels)):
            if int(labels[i][0]) == 0:
                if (isDS):
                    print(testSentencesDS[i])
                else:
                    print(testSentences[i])

        # evaluate the model
        print('\nEvaluating the model...')
        count = 0
        if (isDS):
            for i in range(len(testSentencesDS) - 1):
                if testLabelsDS[i] == int(labels[i][0]):
                    count += 1
            print('Accuracy at 1: ', count/len(testSentencesDS))
        else:
            for i in range(len(testSentences) - 1):
                if int(labels[i][0]) == 0 and testLabels[i] == int(labels[i][0]):
                    count += 1
            print('Accuracy at 1: ', count / len(testSentences))
        print('Precision at 1:', result.precision)
        print('Recall at 1:', result.recall)
        print('Total number of examples:', result.nexamples)
        print('Number of correctly predicted examples:', count, '\n')

        # classify sample texts
        print('Classifying sample texts...')
        texts = ['homework google', 'try it yourself', 'you didnt show any effort']
        labels = classifier.predict(texts)
        print(texts)
        print(labels)

file = '../Models/LSTM/labeled_document_seconditer.json'
isDS = False
instance = FastText()
instance.classify(file,isDS)
# import os
# from fasttuning.genetics import *
# HOME = os.path.expanduser("~")
# params = dict()
# params['epoch'] = [1, 5, 10, 15, 20, 25, 30, 35]
# params['lr'] = [i/10 for i in range(1, 11)]
# params['min_count'] = [1, 10, 20]
# params['word_ngrams'] = [1, 2, 3]
#
# n_individuals = 5
# n_rounds = 2
# good_label = '0'
# train_file = "training_seconditer.txt"
# test_file = "testing_seconditer.txt"
# model_name = 'model_temp'
#
# print('Train file : ', train_file)
# print('Test file : ', test_file)
#
# experience = Experience(params,
#                         n_individuals,
#                         n_rounds=n_rounds,
#                         good_label=good_label,
#                         train_file=train_file,
#                         test_file=test_file)
#
# individuals = experience.launch()
# print(individuals)
# idx = 1
# for i in individuals:
#     model = ("model_%d", idx)
#     idx += 1
#     i.save(model)
#     classifier = fasttext.supervised(train_file, model)
#     result = classifier.test(test_file)
#     metrics = get_metrics(classifier, test_file, good_label)
#     print(metrics['accuracy'])
#     print(metrics['recall'])
#     print(metrics['confusion_matrix'])
#     print(metrics['classes'])
#     print(metrics['examples'])
#     print('Precision at 1:', result.precision)
#     print('Recall at 1:', result.recall)
#     print('Total number of examples:', result.nexamples)

# temp_file = 'temp_model_genetics.bin'
# if os.path.isfile(temp_file):
#     os.remove(temp_file)
