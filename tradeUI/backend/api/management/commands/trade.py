from api.models import Trade
from django.core.management.base import BaseCommand
from django.utils import timezone

from main_loop import trade


class Command(BaseCommand):
    help = ''

    @classmethod
    def init(cls):
        if Trade.objects.count() == 0:
            Trade.objects.create(amount=1000, open_rate=1, close_date=timezone.now(), close_rate=1)
        return cls.balance()

    @staticmethod
    def balance():
        t = Trade.objects.order_by('-id').first()
        close_amount = t.close_rate * t.amount if t.close_date else None
        return (0, close_amount) if close_amount else (t.amount, 0)

    @staticmethod
    def buy(qty, rate):
        Trade.objects.create(amount=qty, open_rate=rate)

    @staticmethod
    def sell(rate):
        Trade.objects.order_by('-id').first().close(rate)

    def handle(self, *args, **options):
        trade(self)
