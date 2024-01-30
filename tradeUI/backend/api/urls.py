from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import *

router = DefaultRouter()
router.register('trade', TradeViewSet, basename='trade')

urlpatterns = [
    path('close/<int:pk>', CloseView.as_view(), name='close'),
    path('', include(router.urls)),
]
