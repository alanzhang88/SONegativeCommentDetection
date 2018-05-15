var express = require('express');
var router = express.Router();
var postNum = 1;



//Currently hard coding collection name; ideally take as input

router.get('/', function(req, res, next) {
  const db = req.app.locals.db;
  let posts = db.collection('PostFirstIter');

  var startId = req.query.postStartID;
  var scoreThreshold = req.query.score;
  var sortScore = req.query.sortScore;
  var sortID = req.query.sortID;
  var zeroComments = req.query.zeroComments;


  //create index
  posts.createIndex('Id', function(err){
    if (err) throw err;
  });

  //if sort post id is chosen 

  if (sortID) {
    posts.find({Id:{'$gte':startId}, Score:{'$lte':scoreThreshold}}).sort({Id : 1}).limit(1000).toArray((err, result) => {
      if (err) throw err;
      let len = result.length;
      var postCollection = [];
  
      var morePost = (len === (postNum + 1));
      var nextPostId = morePost ? result[postNum].postid : -1;
      var partial = morePost ? result.slice(0, postNum) : result;
  
      for(let i = 0; i < partial.length; i++){
        var singlePost = {};
        singlePost.Id = result[i].Id;
        singlePost.Score = result[i].Score;
        singlePost.commentCount = result[i].commentCount;
        singlePost.Body = result[i].Body;
        singlePost.Comments = result[i].Comments;
  
        postCollection.push(singlePost);
      }
  
      res.render('post', {posts:postCollection, nextId: nextPostId});
    })

  }else if(sortScore){
    posts.find({Id:{'$gte':startId}, Score:{'$lte':scoreThreshold}}).sort({score : -1}).limit(1000).toArray((err, result) => {
      if (err) throw err;
      let len = result.length;
      var postCollection = [];
  
      var morePost = (len === (postNum + 1));
      var nextPostId = morePost ? result[postNum].postid : -1;
      var partial = morePost ? result.slice(0, postNum) : result;
  
      for(let i = 0; i < partial.length; i++){
        var singlePost = {};
        singlePost.Id = result[i].Id;
        singlePost.Score = result[i].Score;
        singlePost.commentCount = result[i].commentCount;
        singlePost.Body = result[i].Body;
        singlePost.Comments = result[i].Comments;
  
        postCollection.push(singlePost);
      }
  
      res.render('post', {posts:postCollection, nextId: nextPostId});
    })

  }

  
});

module.exports = router;
