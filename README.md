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
models/ <br />
--LSTM <br />
* LSTM.py: build and train LSTM model <br />
* LSTM_tuning.py: tune hyperparameters for single stack LSTM <br />
* LSTM_tuning_stack2.py: tune hyperparameters for 2-stack LSTM <br />
* LSTM_tuning_stack3.py: tune hyperparameters for 3-stack LSTM <br />

--CNN <br />
* embeddings/: preprocess data to be used for models <br />
* CNNutil.py: build and train CNN model <br />
* run_experiments.sh: script for hyperparameter tuning <br />
* plot_graph.ipynb: jupyter notebook for graph plotting <br />
* test_Param.py: dump different parameter values from experiments to JSON file <br />

--FastText <br />
* FastText.py: build and train FastText model <br />
* FastTextTuning.ipynb: jupyter notebook for FastText hyperparameters plotting <br />

--DataExploration <br />
* dataprocessing.ipynb: Cleanup, extract and merge labeled data
* datavisulization.ipynb: Apply classification and based on word importance, visulize the top 100 words
* prediction_evalution.ipynb: Build structure for tuning data processing and plotting

dataExtraction/: XML parser, extract data from SO data dump <br />
demoWeb/: UI for negative comments prediction web app       <br />
predictWeb/: RESTFul API, with models deployed on the server. Please refer to README.md inside the folder for usage. <br />








