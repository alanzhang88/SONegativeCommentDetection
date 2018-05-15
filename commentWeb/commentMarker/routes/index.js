var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Comment Labeling Options',
                        dbURI: req.cookies.dbURI ? req.cookies.dbURI : 'mongodb://cs230:1234@ds014648.mlab.com:14648/cs230db3',
                        dbName: req.cookies.dbName ? req.cookies.dbName : 'cs230db3',
                        collectionName: req.cookies.collectionName ? req.cookies.collectionName : 'PostFirstIter'
                      });
});

module.exports = router;
