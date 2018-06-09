import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';

import { SearchFormComponent } from './search-form/search-form.component';
import { CommentsDisplayComponent } from './comments-display/comments-display.component';
import { UserDisplayComponent } from './user-display/user-display.component';

const appRoutes: Routes = [
    {'path':'', 'component': SearchFormComponent},
    {'path':'comments', 'component': CommentsDisplayComponent},
    {'path':'user', 'component': UserDisplayComponent}
];

@NgModule({
    imports: [ RouterModule.forRoot(appRoutes) ],
    exports: [ RouterModule ]
})

export class AppRoutingModule{}