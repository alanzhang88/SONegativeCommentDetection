# Project Demo

A barebones Django app, which can easily be deployed to Heroku.

## Folder Structure
* manage.py----puts package on sys.path abd sets the DJANGO_SETTINGS_MODULE environment variable so that it points to settings.py file.
* nltk.txt----data needs to be downloaded from nltk database
* Pipfile, Pipfil.lock----management of dependencies
* Procfile, Procfile.windows----explicitly declare app process types and entry points
* staticfiles/----collects static files from app 
* demo/----demo app folder 
  - admin.py----reads metadata from models, automatic admin interface
  - Classify.py----classifies 3 models to predict the labels of input 
  - migrations/----propagates changes make to models
  - settings.py----contains configuration of app
  - static/----static files
  - templates/----templates of views
  - tests.py----designs tests for app
  - urls.py----designs URLs for app
  - views.py----contains all app views
  - wsgi.py----entry point of the app, WSGI is a python standard for Web-applications and means Web Server Gateway Interface
  - Models/----
    - FastText/----FastText Model
	- CNN/----CNN Model
	- LSTM/----LSTM Model

## Usage

The homepage of API (https://predictlabel.herokuapp.com/) does not have any UI. In order to predict negativity of text, a JSON request needs to be sent to the API url path where "\classify" is appended to the end (https://predictlabel.herokuapp.com/classify). If you just visit the url without the request, there will be an error. You can use Postman to test the API. <br/>

The API takes a  JSON request with the list of comments as input and returns the JSON response with positive and negative classes possibilities as output. Postman can be used to test the API. <br/>

Sample Postman Request: 
* type: Post method
* path: 
  - https://predictlabel.herokuapp.com/classify (if test the remote server, this has already be deployed, you can directly test it using Postman)
  - https://localhost:5000/classify (if test the local server, follow the Running Locally section)
* headers:
  - key: content-type
  - vaue: application-json
* body: check raw tab, copy and paste this --> {"comments": ["What error are you getting?", "do your homework"]} <-- into the text box
* click send button

Sample Postman Response:
* {"lstm_labels": [[0.001398136024363339, 0.9986018538475037], [0.9758431911468506, 0.024156853556632996]], "cnn_labels": [[0.7910102605819702, 0.20898973941802979], [0.7584044933319092, 0.24159550666809082]], "fasttext_labels": [[0.009766000000000052, 0.990234], [0.998047, 0.0019529999999999825]]}
* the first element in the tuple is the negative label possibily, the second element is the positive label possibility 


## Running Locally

Make sure you have Python [installed properly](http://install.python-guide.org). Also, install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and [Postgres](https://devcenter.heroku.com/articles/heroku-postgresql#local-setup).

```sh
$ cd predictWeb
$ pipenv install
$ pipenv shell
$ python manage.py migrate
$ python manage.py runserver 0.0.0.0:5000
```
If you visit http://localhost:5000 and should see hello world. Go to Postman and folow the previous session to test. The only that needs to be changed is that the path is

Your should now be able to run the demo on [localhost:5000](http://localhost:5000/).

## Deploying to Heroku

```sh
$ heroku create
$ git push heroku master

$ heroku run python manage.py migrate
$ heroku open
```
or

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)