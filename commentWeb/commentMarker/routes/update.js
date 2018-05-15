var express = require('express');
var router = express.Router();


router.get('/',(req,res)=>{
  let db = req.app.locals.db;
  // console.log(db);
  
});

module.exports = router;
