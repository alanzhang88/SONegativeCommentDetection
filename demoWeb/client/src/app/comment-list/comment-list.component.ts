import { Component, OnInit, Input } from '@angular/core';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-comment-list',
  templateUrl: './comment-list.component.html',
  styleUrls: ['./comment-list.component.css']
})
export class CommentListComponent implements OnInit {

  @Input() commentList = null;
  @Input() modelSelection = null;

  constructor(private sanitizer:DomSanitizer) { }

  ngOnInit() {
  }


  getStyle(value){
    return this.sanitizer.bypassSecurityTrustStyle(`width: ${Math.round(value*100)}%`);
  }

  getHtml(value){
    return this.sanitizer.bypassSecurityTrustHtml(value);
  }

  getValue(value){
    return Math.round(value*100);
  }


}
