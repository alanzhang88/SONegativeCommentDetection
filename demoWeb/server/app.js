var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');

const apiRouter = require('./routes/api');

var app = express();

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());


app.use('/api',apiRouter);
app.use(express.static(path.join(__dirname, 'public')));
app.get('*',(req,res)=>{
    res.sendFile(path.join(__dirname, 'public')+'/index.html');
});

module.exports = app;
