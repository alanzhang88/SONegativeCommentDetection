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
    trainTexts = []
    testTexts = []
    trainSentences = []

    def __init__(self):
        print("init")

    def preprocessData(self):
        trainTexts = []
        testTexts = []
        trainSentences = []
        print("Loading and preprocessing data...\n")
        # load training and testing data
        with open('../LSTM/labeled_document_firstiter.json') as json_data:
            allTrainData = json.load(json_data)

        with open('../LSTM/labeled_document_seconditer.json') as json_data:
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
        for phrase in trainPhrases:
            if not isinstance(phrase, str):
                continue
            tokens = word_tokenize(phrase)
            parsedWords = []
            for t in tokens:
                if t not in stopWords:
                    parsedWords.append(engStemmer.stem(t))
            trainTexts.append(parsedWords)

        for phrase in testPhrases:
            if not isinstance(phrase, str):
                continue
            trainSentences.append(phrase)
            tokens = word_tokenize(phrase)
            parsedWords = []
            for t in tokens:
                if t not in stopWords:
                    parsedWords.append(engStemmer.stem(t))
            testTexts.append(parsedWords)
        return (trainTexts, testTexts, trainLabel, testLabel)

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

    def classify(self,isDS):
        # preprocess data
        (trainTexts, testTexts, trainLabels, testLabels) = self.preprocessData()
        # create training and testing file
        if (isDS):
            trainSentencesDS, trainLabelsDS = self.downsampling(trainTexts, trainLabels, 0)
            self.outputSentencesToFile("./data/trainingDS_all", trainSentencesDS, trainLabelsDS)
            testSentencesDS, testLabelsDS = self.downsampling(testTexts, testLabels, 0)
            self.outputSentencesToFile("./data/testingDS_all", testSentencesDS, testLabelsDS)
        else:
            self.outputPhrasesToFile("./data/training_all", trainTexts, trainLabels)
            self.outputPhrasesToFile("./data/testing_all", testTexts, testLabels)
            testSentences = self.extractText(testTexts)
        # train the fasttext model
        if (isDS):
            #   downsampling inbalanced data
            classifier = fasttext.supervised('./data/trainingDS_all.txt', './model/best_model')
            result = classifier.test('./data/testingDS_all.txt')
        else:
            #   without downsampling inbalanced data
            classifier = fasttext.supervised('./data/training_all.txt', './model/best_model', epoch=5, lr=0.1, dim=300,
                                             word_ngrams=1, loss='ns', ws=1, min_count=5, bucket=2000000)
            result = classifier.test('./data/testing_all.txt')

        # (trainTexts, testTexts, trainLabels, testLabels) = self.preprocessData()
        # self.outputPhrasesToFile("./data/training_all", trainTexts, trainLabels)
        # self.outputPhrasesToFile("./data/testing_all", testTexts, testLabels)
        # testSentences = self.extractText(testTexts)
        # classifier = fasttext.supervised('./data/training_all.txt', './model/best_model', epoch=5, lr=0.1, dim=300,
                                        # word_ngrams=1, loss='ns', ws=1, min_count=5, bucket=2000000)
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
                if testLabels[i] == int(labels[i][0]):
                    count += 1
            print('Accuracy at 1: ', count / len(testSentences))
        print('Precision at 1:', result.precision)
        print('Recall at 1:', result.recall)
        print('Total number of examples:', result.nexamples)
        print('Number of correctly predicted examples:', count, '\n')

        # classify sample texts
        print('Classifying sample texts...')
        texts = ['If you intend to become a professional programmer, you are going to have to learn to look up documentation. And to run programs if you want to know what happens when you run them. Your mother is not always going to be on hand to spoon-feed you your breakfast.', 'try it yourself', 'you didnt show any effort']
        labels = classifier.predict(texts)
        print(texts)
        print(labels)

isDS = False
instance = FastText()
instance.classify(isDS)


