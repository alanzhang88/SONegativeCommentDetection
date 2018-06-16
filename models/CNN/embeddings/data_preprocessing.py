import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize.casual import remove_handles
import re
import string
import csv
import json

import argparse


def preprocess(commentList):
    translator = str.maketrans('', '', string.punctuation)
    with open('processed.csv', 'w') as csvfile:
        fieldnames = ['comments', 'labels']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for item in commentList:
            comment = remove_handles(item["comment"]).translate(translator)
            csvwriter.writerow({'comments': comment, 'labels': item["label"]})



def load_csvfile(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, ["comment", "label"])
        commentsList = []
        for row in reader:
            items = {}
            items['comment'] = row["comment"].lower()
            items['label'] = row["label"]
            commentsList.append(items)
    print (len(commentsList))
    preprocess(commentsList)


# def load_json(filename):
#     with open(filename) as json_data:
#         data = json.load(json_data)
#     commentList = []
#     count=0
#     for comments in data['Comment']:
#         items = {}
#         items['comment'] = comments.lower()
#         items['label']  = data['CommentLabel'][count]
#         count += 1
#         commentList.append(items)
#     preprocess(commentList)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv',dest='csvfilename',type=str,required=False)
    args = parser.parse_args()

    if args.csvfilename is not None:
        load_csvfile(args.csvfilename)
    
   
