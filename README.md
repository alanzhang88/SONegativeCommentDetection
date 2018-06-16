# SONegativeCommentDetection

This project aims at finding out negative comments from StackOverflow data using NLP models. We train data with LSTM, CNN and FastText.

## Basic Information
We use keras and tensorflow to train our model over more than 8000 comments which are labelled by ourselves.

## Labelled Data
The labelled data can be found in the Data folder. We decide to release the data so that anyone who is interested in sentiment analysis on StackOverflow comments can make use of our data.

## Deployment

Demo server is deployed live on Heroku:
https://sonegativecommentdetection.herokuapp.com/

## Code Structures:
Models/ 
--LSTM
--CNN
--FastText

dataExtraction/: XML parser, extract data from SO data dump 
demoWeb/: UI for negative comments prediction web app
predictWeb/: RESTFul API, with models deployed on the server






