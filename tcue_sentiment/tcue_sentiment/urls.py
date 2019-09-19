"""tcue_sentiment URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from tcue_classifier.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),

    path('sentiment/', sentiment, name='sentiment'),

    path('sentiment/classification/', sentiment_classification, name='sentiment_classification'),
    path('sentiment/classification/positive/', sentiment_positive, name='sentiment_positive'),
    path('sentiment/classification/neutral/', sentiment_neutral, name='sentiment_neutral'),
    path('sentiment/classification/negative/', sentiment_negative, name='sentiment_negative'),
    path('sentiment/classification/none/', sentiment_none, name='sentiment_none'),
    path('sentiment/classification/refresh/', sentiment_refresh, name='sentiment_refresh'),
]
