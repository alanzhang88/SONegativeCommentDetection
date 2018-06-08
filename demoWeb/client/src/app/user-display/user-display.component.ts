import { Component, OnInit, OnDestroy } from '@angular/core';

import { CommentClassifyService } from '../comment-classify.service';

@Component({
  selector: 'app-user-display',
  templateUrl: './user-display.component.html',
  styleUrls: ['./user-display.component.css']
})
export class UserDisplayComponent implements OnInit, OnDestroy {

  commentList = null;
  modelSelection:string = 'lstm_labels';

  constructor(private commentClassifyService: CommentClassifyService) { }

  ngOnInit() {
    this.commentList = this.commentClassifyService.getList();
  }

  changeModel(name:string){
    this.modelSelection = name;
    this.commentClassifyService.sortOrder(name);
  }

  changeActive(name:string){
    if(name === this.modelSelection){
      return "nav-link active";
    }
    else return "nav-link";
  }

  ngOnDestroy(){
    this.commentClassifyService.clearList();
  }

}
