var express = require('express');
var router = express.Router();
// var postNum = 1;
// var cursor = null;


router.use(function(req,res,next){
  if(req.query.postStartID&&req.query.score){
    // const db = req.app.locals.db;
    // let posts = req.app.locals.collection;
    const db = req.app.locals.userData[req.cookies.clientId].db;
    let posts = req.app.locals.userData[req.cookies.clientId].collection;
    let cursor = null;

    const startId = +req.query.postStartID;
    const scoreThreshold = +req.query.score;
    var sortScore = req.query.sortScore ? "True" : "False";
    var sortID = req.query.sortID ? "True" : "False";
    var zeroComments = req.query.zeroComments ? "True" : "False";
    var model = req.query.model ? true : false;
    let minComment = zeroComments === 'True' ? 0 : 1

    //create index
    posts.createIndex('Id', function(err){
      if (err) throw err;
    });
    posts.createIndex('Score', function(err){
      if (err) throw err;
    });

    let queryObj = { "Id":{$gte:startId},
                     "Score": {$lte:scoreThreshold},
                     'CommentCount':{'$gte':minComment},
                     'CommentsLabel':{'$exists':false}
                   };
    if(model){
      queryObj['FirstIterCommentsLabel'] = {'$exists':true};
    }

    if(sortID === "True"){
      cursor = posts.find(queryObj).sort({Id : 1}).limit(1000);
    }else if(sortScore === "True"){
      cursor = posts.find(queryObj).sort({Score: 1}).limit(1000);
    }
    else{
      cursor = posts.aggregate([{'$match':queryObj},{'$sample':{'size':1000}}]);
    }
    req.app.locals.userData[req.cookies.clientId].cursor = cursor;
  }
  next();
});

router.get('/', function(req, res, next) {
  let cursor = req.app.locals.userData[req.cookies.clientId].cursor;
  cursor.hasNext().then(
    (result) =>{
      if(result){
        return cursor.next();
      }
      else{
        return res.send('No more Post to label');
      }
    }
  ).then(
    (result) => {
      if(result == null){
        console.log("NULL");
      }
      // console.log(result);
      // var postCollection = [];
      var singlePost = {};
      singlePost.Id = result['Id'];
      singlePost.Score = result['Score'];
      singlePost.commentCount = result['CommentCount'];
      singlePost.Body = result['Body'];
      singlePost.Comments = result['Comments'];
      // console.log(singlePost);
      // postCollection.push(singlePost);
      res.render('display', {post: singlePost});
    }
  ).catch(
    (err) => {
      console.log(err);
    }
  );
  // const db = req.app.locals.db;
  // let posts = req.app.locals.collection;
  //
  // const startId = +req.query.postStartID;
  // const scoreThreshold = +req.query.score;
  // var sortScore = req.query.sortScore;
  // var sortID = req.query.sortID;
  // var zeroComments = req.query.zeroComments;
  //
  // //create index
  // posts.createIndex('Id', function(err){
  //   if (err) throw err;
  // });
  //
  // if(sortID === "True"){
  //   cursor = posts.find({"Id":{$gte:startId}, "Score": {$lte:scoreThreshold}}).sort({Id : 1}).limit(1000);
  //
  //   cursor.hasNext((err, arr) => {
  //    if(err) throw err;
  //    var hasNext = 1;
  //     cursor.next(function(err, result){
  //       //var hasNext = cursor.hasNext() ? 1 : -1;
  //       if (err) throw err;
  //       if(result == null){
  //         console.log("NULL");
  //       }
  //       var postCollection = [];
  //       var singlePost = {};
  //       singlePost.Id = result['Id'];
  //       singlePost.Score = result['Score'];
  //       singlePost.commentCount = result['commentCount'];
  //       singlePost.Body = result['Body'];
  //       singlePost.Comments = result['Comments'];
  //       postCollection.push(singlePost);
  //       res.render('display', {post: postCollection, nextPost: hasNext});
  //     });
  //  });
  // }else if(sortScore === "True"){
  //   cursor = posts.find({Id:{'$gte':startId}, Score: {'$lte':scoreThreshold}}).sort({Score: -1}).limit(1000);
  //
  //   cursor.hasNext((err, arr) => {
  //     if(err) throw err;
  //     var hasNext = 1;
  //      cursor.next(function(err, result){
  //        //var hasNext = cursor.hasNext() ? 1 : -1;
  //        if (err) throw err;
  //        if(result == null){
  //          console.log("NULL");
  //        }
  //        var postCollection = [];
  //        var singlePost = {};
  //        singlePost.Id = result['Id'];
  //        singlePost.Score = result['Score'];
  //        singlePost.commentCount = result['commentCount'];
  //        singlePost.Body = result['Body'];
  //        singlePost.Comments = result['Comments'];
  //        postCollection.push(singlePost);
  //        res.render('display', {post: postCollection, nextPost: hasNext});
  //      });
  //
  //   });
  // }


});

module.exports = router;
