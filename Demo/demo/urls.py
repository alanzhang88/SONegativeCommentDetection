from django.conf.urls import include, url
from django.urls import path
from django.contrib import admin
admin.autodiscover()
import demo.views

# Examples:
# url(r'^$', 'gettingstarted.views.home', name='home'),
# url(r'^blog/', include('blog.urls')),

urlpatterns = [
    url(r'^$', demo.views.classify, name='demo'),
    path('admin/', admin.site.urls),
]
