from django.urls import path
from api import views
from django.views.decorators.csrf import csrf_exempt


urlpatterns = [
    path('pest_detection/',
         csrf_exempt(views.pest_model_prediction)),
    path('crop_disease_detection/', csrf_exempt(views.crop_model_pred)),
    path('crop_recomendation_simple/',
         csrf_exempt(views.crop_simple_recomendation_prediction)),
    path('crop_recomendation_advance/',
         csrf_exempt(views.crop_advance_recomendation_prediction)),
    path('fertilizer_recomendation/',
         csrf_exempt(views.fertilizer_recomendation_prediction)),
]
