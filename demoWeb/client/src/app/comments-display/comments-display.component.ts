import { Component, OnInit, OnDestroy } from '@angular/core';
import { NgForm } from '@angular/forms';
import { CommentClassifyService } from '../comment-classify.service';

@Component({
  selector: 'app-comments-display',
  templateUrl: './comments-display.component.html',
  styleUrls: ['./comments-display.component.css']
})
export class CommentsDisplayComponent implements OnInit, OnDestroy {

  comment:string = null;
  commentList = null;
  modelSelection:string = 'lstm_labels';

  constructor(private commentClassifyService: CommentClassifyService) { }

  ngOnInit() {
    this.commentList = this.commentClassifyService.getList();
  }

  onSubmit(form:NgForm){
    // console.log(form.value.comment);
    this.commentClassifyService.appendComment(form.value.comment);
    form.reset();
  }

  changeModel(name:string){
    this.modelSelection = name;
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
