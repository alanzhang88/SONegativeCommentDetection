import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize.casual import remove_handles
import re
import string
import csv


# def preprocess(commentList):
#     comment = ','.join(commentList).lower()
#     tokenizer = RegexpTokenizer(r'\w+')
#     tokens = tokenizer.tokenize(comment)
#     # filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
#     with open("preprocessed.csv", "w") as f:
#         writer = csv.writer(f, delimiter=' ')
#         for item in tokens:
#             writer.writerow(item)



# def load_file(filename):
#     with open(filename, newline='') as csvfile:
#         reader = csv.DictReader(csvfile, ["comment", "label"])
#         comments = []
#         for row in reader:
#             comments.append(row["comment"])
#     print (comments)
#     preprocess(comments)

def preprocess(commentList):
    translator = str.maketrans('', '', string.punctuation)
    with open('processed.csv', 'w') as csvfile:
        fieldnames = ['comments', 'labels']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for item in commentList:
            comment = remove_handles(item["comment"]).translate(translator)
            csvwriter.writerow({'comments': comment, 'labels': item["label"]})





def load_file(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, ["comment", "label"])
        commentsList = []
        for row in reader:
            items = {}
            items['comment'] = row["comment"].lower()
            items['label'] = row["label"]
            commentsList.append(items)
    preprocess(commentsList)

load_file("./labeled_comments.csv")
