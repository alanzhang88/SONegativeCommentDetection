import fasttext

# # Skipgram model
# model = fasttext.skipgram('data.txt', 'model')
# print(model.words) # list of words in dictionary
#
# # CBOW model
# model = fasttext.cbow('data.txt', 'model')
# print(model.words) # list of words in dictionary

classifier = fasttext.supervised('train.txt', 'model')
result = classifier.test('test.txt')
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)
