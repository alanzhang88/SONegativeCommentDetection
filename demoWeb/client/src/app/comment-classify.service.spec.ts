import { TestBed, inject } from '@angular/core/testing';

import { CommentClassifyService } from './comment-classify.service';

describe('CommentClassifyService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [CommentClassifyService]
    });
  });

  it('should be created', inject([CommentClassifyService], (service: CommentClassifyService) => {
    expect(service).toBeTruthy();
  }));
});
