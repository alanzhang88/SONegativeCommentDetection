var express = require('express');
var router = express.Router();
var postNum = 1;
var cursor = null;


router.get('/', function(req, res, next) {
  const db = req.app.locals.db;
  let posts = req.app.locals.collection;

  const startId = +req.query.postStartID;
  const scoreThreshold = +req.query.score;
  var sortScore = req.query.sortScore;
  var sortID = req.query.sortID;
  var zeroComments = req.query.zeroComments;
  
  //create index
  posts.createIndex('Id', function(err){
    if (err) throw err;
  });

  if(sortID === "True"){
    cursor = posts.find({"Id":{$gte:startId}, "Score": {$lte:scoreThreshold}}).sort({Id : 1}).limit(1000);
    cursor.next(function(err, result){
      if (err) throw err;
      if(result == null){
        console.log("NULL");
      }
      var postCollection = [];
      var singlePost = {};
      singlePost.Id = result['Id'];
      singlePost.Score = result['Score'];
      singlePost.commentCount = result['commentCount'];
      singlePost.Body = result['Body'];
      singlePost.Comments = result['Comments'];
      postCollection.push(singlePost);
      res.render('display', {post: postCollection});
    });
    
  }else if(sortScore === "True"){
    cursor = posts.find({Id:{'$gte':startId}, Score: {'$lte':scoreThreshold}}).sort({Score: -1}).limit(1000);
    cursor.next(function(err, result){
      if (err) throw err;
      if(result == null){
        console.log("NULL");
      }
      var postCollection = [];
      var singlePost = {};
      singlePost.Id = result['Id'];
      singlePost.Score = result['Score'];
      singlePost.commentCount = result['commentCount'];
      singlePost.Body = result['Body'];
      singlePost.Comments = result['Comments'];
      postCollection.push(singlePost);
      res.render('display', {post: postCollection});
    });
  }
});

module.exports = router;
