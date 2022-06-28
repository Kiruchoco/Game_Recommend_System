from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),
    path('testpage',views.testpage,name='testpage'),
    path('index',views.index,name='index'),
    #path('test_input',views.input_test,name='input_test'),
    path('first',views.first,name='first'),
    path('second',views.second,name='second'),
]