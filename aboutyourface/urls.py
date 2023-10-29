from django.urls import path
from aboutyourface_app import views

urlpatterns = [
    path('face_detection/', views.face_detection, name='face_detection'),
]
