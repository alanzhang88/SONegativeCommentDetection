var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
var assert = require('assert');
var uniqid = require('uniqid');
var index = require('./routes/index');
var display = require('./routes/display');
var update = require('./routes/update');

var app = express();

//db configs
//const dbName = "CS230DB";
//const url ='mongodb://cs230:1234@ds014648.mlab.com:14648/cs230db3';

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
      let clientId = req.cookies.clientId ? req.cookies.clientId : uniqid();
      req.cookies.clientId = clientId;
      console.log(req.cookies.clientId);
      if(!app.locals.userData){
        app.locals.userData = {};
      }
      // app.locals.db = db;
      // app.locals.collection = db.collection(req.query.collectionName);
      app.locals.userData[clientId] = {db,collection:db.collection(req.query.collectionName)};
      res.cookie('dbURI',req.query.dbURI,{'expires': new Date(Date.now()+172800000)});
      res.cookie('dbName',req.query.dbName,{'expires': new Date(Date.now()+172800000)});
      res.cookie('collectionName',req.query.collectionName,{'expires': new Date(Date.now()+172800000)});
      res.cookie('clientId',clientId,{'expires': new Date(Date.now()+172800000)});
      next();
    });
  }
  else{
    next();
  }
});

app.use(function(req,res,next){
  if(app.locals.userData&&req.cookies.clientId){
    let clientId = req.cookies.clientId;
    if(app.locals.userData[clientId]){
      let timeNow = Date.now();

      Object.keys(app.locals.userData).forEach(function(key,index) {
      // key: the name of the object key
      // index: the ordinal position of the key within the object
        if(app.locals.userData[key].lastVisited && timeNow - app.locals.userData[key].lastVisited >= 172800000){
          console.log(`delete key ${key}`);
          delete app.locals.userData[key];
        }
      });
      app.locals.userData[clientId].lastVisited = timeNow;
    }    
  }
  next();
});

app.use('/', index);
app.use('/display', display);
app.use('/update',update);

module.exports = app;
