var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var assert = require('assert');
var index = require('./routes/index');
var display = require('./routes/display');
var update = require('./routes/update');

var app = express();

//db configs
const dbName = "CS230DB";
const url ='mongodb://cs230:1234@ds014648.mlab.com:14648/cs230db3';

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));


app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  res.header('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS');
  next();
});

//establish connection to mongodb
app.use(function(req, res, next){
  if(req.query.dbURI&&req.query.dbName&&req.query.collectionName){
    const mongoDB = require('mongodb');
    const dbClient = mongoDB.MongoClient;
    dbClient.connect(req.query.dbURI, (err, conn)=>{
      assert.equal(err, null);
      let db = conn.db(req.query.dbName);
      console.log("Connected successful to mongodb");
      app.locals.db = db;
      app.locals.collection = db.collection(req.query.collectionName);
      next();
    });
  }
  else{
    next();
  }
});

app.use('/', index);
app.use('/display', display);
app.use('/update',update);

module.exports = app;
