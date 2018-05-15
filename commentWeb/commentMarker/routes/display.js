var express = require('express');
var router = express.Router();
var postNum = 1;


router.post('/display:startID', function(req, res, next) {
  const db = req.locals.db;
  let posts = db.collection('PostFirstIter');

  //create index
  posts.createIndex('Id', function(err){
    if (err) throw err;
  });

  //if sort post id is chosen 

  posts.find({'Id':{'$gte':args.startId}},{'Score':{'$lte':args.score}},{'CommentCount':{'$gt':commentCount}},{'BodyLabel':{'$exists':False}}).sort('Id',pymongo.ASCENDING).limit(1000).toArray((err, result) => {
    if (err) throw err;
    let len = result.length;


    var postCollection = [];
    for(let i = 0; i < partial.length; i++){
      var singlePost = {};
      singlePost.Id = result[i].Id;
      singlePost.Score = result[i].Score;
      singlePost.commentCount = result[i].commentCount;
      singlePost.Body = result[i].Body;
      singlePost.Comments = result[i].Comments;

      postCollection.push(singlePost);
    }

    res.render('post', {posts:postCollection});
  })
});

module.exports = router;
