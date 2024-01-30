from django.db import models
from django.utils import timezone


class Trade(models.Model):
    amount = models.FloatField()
    open_rate = models.FloatField()
    close_rate = models.FloatField(null=True)
    open_date = models.DateTimeField(auto_now_add=True)
    close_date = models.DateTimeField(null=True)

    def close(self, rate):
        self.close_date = timezone.now()
        self.close_rate = rate
        self.save()

    def __str__(self) -> str:
        return f'({self.id}) Closed {self.close_date}' if self.close_date else f'({self.id}) Opened {self.open_date}'
