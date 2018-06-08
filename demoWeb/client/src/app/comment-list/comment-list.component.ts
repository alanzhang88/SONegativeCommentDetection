import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-comment-list',
  templateUrl: './comment-list.component.html',
  styleUrls: ['./comment-list.component.css']
})
export class CommentListComponent implements OnInit {

  @Input() commentList = null;
  @Input() modelSelection = null;

  constructor() { }

  ngOnInit() {
  }


  getValue(value){
    return Math.round(value*100);
  }


}
