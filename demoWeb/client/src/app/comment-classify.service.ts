import { Injectable } from '@angular/core';
import { Http, Response } from '@angular/http';

@Injectable({
  providedIn: 'root'
})
export class CommentClassifyService {

  commentList = [];

  constructor(private http:Http) { }

  getList(){
    return this.commentList;
  }

  appendComment(comment:string){
    this.http.post('/api/sentence',{
      'comment':comment
    }).subscribe(
      (response:Response) => {
        let data = response.json();
        console.log(data);
        this.commentList.push(data);
      }
    );
  }

  searchUser(userId:string){
    this.http.post(`/api/user/${userId}`,{}).subscribe(
      (response:Response) => {
        let data = response.json();
        console.log(data);
        for(let c of data.items){
          this.commentList.push(c);
        }
        this.sortOrder('lstm_labels');
      }
    );
  }

  sortOrder(modelName:string){
    switch (modelName) {
      case 'lstm_labels':
        this.commentList.sort((a, b) => {
          return b.lstm_labels - a.lstm_labels;
        });
        break;
      case 'cnn_labels':
        this.commentList.sort((a, b) => {
          return b.cnn_labels - a.cnn_labels;
        });
        break;
      case 'fasttext_labels':
        this.commentList.sort((a, b) => {
          return b.fasttext_labels - a.fasttext_labels;
        });
        break;
    }
  }

  clearList(){
    this.commentList.splice(0,this.commentList.length);
  }
}
