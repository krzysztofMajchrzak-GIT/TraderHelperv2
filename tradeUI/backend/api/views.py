import requests

from django.shortcuts import get_object_or_404
from rest_framework import filters, generics, status, viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

from .serializers import *


class TradeViewSet(viewsets.ModelViewSet):
    permission_classes = (AllowAny,)
    serializer_class = TradeSerializer
    queryset = Trade.objects.order_by('-id')


class CloseView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request, pk):
        r = requests.get("https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval=1m&limit=1")
        t = get_object_or_404(Trade, pk=pk)
        t.close(float(r.json()[0][4]))
        return Response()
