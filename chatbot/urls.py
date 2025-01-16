from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('ask/', views.chatbot_ask, name='chatbot_ask'),
]