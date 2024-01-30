import json
import os

from django.urls import reverse
from django.test import TestCase
from rest_framework.test import APIClient
from .models import *


class SOCModelViewSetTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_list_endpoints(self):
        EXCLUDE_ABSTRACT_BASE = {OrgEntityModel, BUOwnableEntityModel, IPAddress}
        for model in SOCModel.__subclasses__():
            if model in EXCLUDE_ABSTRACT_BASE:
                continue
            viewset = model.__name__.lower()
            self.assertTrue(viewset)

            url = reverse(f'{viewset}-list')
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.data, list)


# class NewCaseAPIViewTestCase(TestCase):
#     def setUp(self):
#         self.client = APIClient()
#         Platform.objects.create(name='Other', platform_type='OTHER')
#         org = Organisation.objects.create(name='General')
#         Queue.objects.create(name='General', organisation_fk=org)

#     def test_new_case(self):
#         for file in os.listdir('test_json')[::-1]:
#             with open('test_json/' + file) as f:
#             # with open('tests.json') as f:
#                 data = json.loads(f.read())
#                 for ticket in data:
#                     org = Organisation.objects.get_or_create(name=ticket['organisation'])
#                 self.client.post(reverse('new'), data, 'json')
