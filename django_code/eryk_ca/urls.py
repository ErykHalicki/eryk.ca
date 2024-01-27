from django.urls import path
from . import views

urlpatterns = [
    path('', views.timeline_view, name='eryk_ca'),
    path('about-me', views.about_me, name='about-me'),
]
