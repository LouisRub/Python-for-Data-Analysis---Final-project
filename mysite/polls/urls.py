from django.urls import path

from . import views


urlpatterns = [
    
    path("", views.index1, name="index"),
    path("apply_Model", views.index2, name="index"),
    path('about/', views.about, name='about'),
    path('about2/',views.about2, name='about2'),
    path("apply_Model2",views.index3,name="index"),
    path("interact/",views.interact,name='interact'),
    
]
