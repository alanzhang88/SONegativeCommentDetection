# SONegativeCommentDetection

This project aims at finding out negative comments from StackOverflow data using NLP models. We train data with LSTM, CNN and FastText.

## Basic Information
We use keras and tensorflow to train and test our model on 11,226 comments which are labelled by ourselves.

## Labelled Data
The labelled data can be found in the Data folder. We decide to release the data so that anyone who is interested in sentiment analysis on StackOverflow comments can make use of our data.

## Deployment

Demo server is deployed live on Heroku:
https://sonegativecommentdetection.herokuapp.com/

## Code Structures:
* data/: 11,226 manually labeled comments
* dataExtraction/: XML parser, extract data from SO data dump 
* demoWeb/: Negative comments prediction web app, which calls the RESTFul API in predictWeb to obtain the positive and negative class possibilities of comments. This app has already deployed so you can directly visit https://sonegativecommentdetection.herokuapp.com/ to test the performance of 3 models on this web.
* predictWeb/: RESTFul API with models deployed on the server. Please refer to README.md inside the folder for usage.
* models/ 
  - LSTM 
    - LSTM.py: build and train LSTM model 
	- LSTM_tuning.py: tune hyperparameters for single stack LSTM 
	- LSTM_tuning_stack2.py: tune hyperparameters for 2-stack LSTM 
	- LSTM_tuning_stack3.py: tune hyperparameters for 3-stack LSTM 
  - CNN 
	- embeddings/: preprocess data to be used for models 
	- CNNutil.py: build and train CNN model 
	- run_experiments.sh: script for hyperparameter tuning 
	- plot_graph.ipynb: jupyter notebook for graph plotting 
	- test_Param.py: dump different parameter values from experiments to JSON file 
  - FastText 
	- FastText.py: build and train FastText model 
	- FastTextTuning.ipynb: jupyter notebook for FastText hyperparameters plotting 
	- data/: training and testing data
	- model/: reusable fasttext classifiers
  -	DataExploration 
	- dataprocessing.ipynb: Cleanup, extract and merge labeled data
	- datavisulization.ipynb: Apply classification and based on word importance, visulize the top 100 words
	- prediction_evalution.ipynb: Build structure for tuning data processing and plotting









