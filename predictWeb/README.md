# Project Demo

A barebones Django app, which can easily be deployed to Heroku.

## Folder Structure

demo/----demo app folder
	admin.py----reads metadata from models, automatic admin interface
	Classify.py----classifies 3 models to predict the labels of input
	migrations/----propagates changes make to models
	settings.py----contains configuration of app
	static/----static files
	templates/----templates of views
	tests.py----designs tests for app
	urls.py----designs URLs for app
	views.py----contains all app views
	wsgi.py----entry point of the app, WSGI is a python standard for Web-applications and means Web Server Gateway Interface
	Models/----
			FastText/----FastText Model
			CNN/----CNN Model
			LSTM/----LSTM Model
manage.py----puts package on sys.path abd sets the DJANGO_SETTINGS_MODULE environment variable so that it points to settings.py file.
nltk.txt----data needs to be downloaded from nltk database
Pipfile, Pipfil.lock----management of dependencies
Procfile, Procfile.windows----explicitly declare app process types and entry points
staticfiles/----collects static files from app 

## Running Locally

Make sure you have Python [installed properly](http://install.python-guide.org). Also, install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and [Postgres](https://devcenter.heroku.com/articles/heroku-postgresql#local-setup).

```sh
<!-- $ git clone git@github.com:heroku/python-getting-started.git
$ cd python-getting-started -->
$ cd predictWeb
$ pipenv install
<!-- $ createdb python_getting_started -->
$ python manage.py migrate
$ python manage.py collectstatic
$ python manage.py runserver 0.0.0.0:5000

<!-- $ heroku local -->
```

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