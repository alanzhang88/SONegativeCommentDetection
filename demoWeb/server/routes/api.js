var express = require('express');
var router = express.Router();
const axios = require('axios');
const decode = require('unescape');
axios.defaults.timeout = 150000;
BACKEND_URL = 'https://predictlabel.herokuapp.com/classify';

function getProb(comments){
    return axios.post(BACKEND_URL,{
        'comments': comments
    }).then(
        (results) => {
            return results.data;
        }
    );
}

router.post("/sentence",(req,res)=>{
    // console.log(req.body.comments);
    let respObj = {'comment': req.body.comment};
    getProb([req.body.comment]).then(
        (results) => {
            for(let key in results){
                respObj[key] = results[key][0][0];
            }
            res.json(respObj);
        }
    ).catch(
        (err) => {
            console.log(err);
        }
    );
});

router.post("/user/:id",(req,res)=>{
    let userId = req.params.id;
    let requestUrl = `https://api.stackexchange.com/2.2/users/${userId}/comments?filter=!-09UyPwUHA4s&order=desc&sort=creation&site=stackoverflow&key=Cb45fgMq3YHzi4EMv*7HuA((`;
    let respObj = {'items':[]};
    axios.get(requestUrl).then(
        (results) => {
            if(!results.data)return res.json();
            let data = results.data.items;
            let commentLists = [];
            for(let i = 0; i < data.length; i++){
                let decodedstr = decode(data[i].body);
                commentLists.push(decodedstr);
                respObj.items.push({'comment':decodedstr});
            }
            // res.json({"comments":commentLists});
            // respObj.comments = commentLists;
            return getProb(commentLists);
        }
    ).then(
        (results) => {
            for(let key in results){
                // respObj[key] = results[key];
                for(let i = 0; i < results[key].length; i++){
                    respObj.items[i][key] = results[key][i][0];
                }
            }
            res.json(respObj);
        }
    ).catch(
        (err) => {
            console.log(err);
        }
    );

});

module.exports = router;