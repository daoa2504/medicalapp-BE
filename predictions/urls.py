from django.urls import path
from . import views
from .views import prediction_view
from .api_views import predict_api, model_info_api, health_check
urlpatterns =[
    path('', views.home),
    path('predict/', prediction_view, name='prediction_view'),

    path('api/predict/', predict_api, name='predict_api'),
    path('api/models/', model_info_api, name='model_info'),
    path('api/health/', health_check, name='health_check'),
]