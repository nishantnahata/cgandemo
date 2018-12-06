from django.urls import path

from . import views

urlpatterns = [
    path('', views.Demo.as_view(), name='demo'),
]
