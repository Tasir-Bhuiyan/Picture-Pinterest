from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='upload'),
    path('compare/', views.compare, name='compare'),
    path('results/', views.results, name='results'),
    path('select-image/', views.select_image, name='select_image'),
    path('reset/', views.reset, name='reset'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
