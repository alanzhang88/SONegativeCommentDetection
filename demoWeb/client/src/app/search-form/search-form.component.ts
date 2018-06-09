import { Component, OnInit } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Router } from '@angular/router';
import { CommentClassifyService } from '../comment-classify.service';

@Component({
  selector: 'app-search-form',
  templateUrl: './search-form.component.html',
  styleUrls: ['./search-form.component.css']
})
export class SearchFormComponent implements OnInit {

  userId:string = null;
  comment:string = null;
  searchUser:boolean = false;

  constructor(private router:Router, private commentClassifyService: CommentClassifyService) { }

  ngOnInit() {
  }

  onSwitch(){
    this.searchUser = !this.searchUser;
  }

  onSubmit(form:NgForm){
    if(this.searchUser){
      //call service on user
      // console.log(form.value.userId);
      this.commentClassifyService.searchUser(form.value.userId);
      this.router.navigate(['user']);
    }
    else{
      //call service on comment
      // console.log(form.value.comment);
      this.commentClassifyService.appendComment(form.value.comment);
      this.router.navigate(['comments']);
    }
  }
}
