var express = require('express');
var router = express.Router();


router.post('/',(req,res)=>{
  let db = req.app.locals.db;
  let collection = req.app.locals.collection;
  if(!collection){
    return res.send('Something Wrong with DB connection');
  }
  // console.log(req.app.locals);
  // console.log(db);
  // console.log(req.body);
  let bodylabel = null
  let Id = parseInt(req.body.Id);
  let commentCount = parseInt(req.body.commentCount);
  let comments = [];
  if(req.body.bodylabel){
    bodylabel = parseInt(req.body.bodylabel);
  }
  for(let i = 0; i < commentCount; i++){
    comments.push(parseInt(req.body[`comment${i+1}`]));
  }
  // console.log(comments);

  collection.findOneAndUpdate({'Id':Id},{'$set':{'CommentsLabel':comments,'BodyLabel':bodylabel}}).then(
    () => {
      console.log(`Success update on postId ${Id}`);
    }
  ).catch((err)=>{
    console.log(err);
  });

  res.redirect('/display');
});

module.exports = router;
