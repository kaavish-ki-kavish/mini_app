from django.shortcuts import render

from django.contrib.auth import get_user_model, logout
from django.core.exceptions import ImproperlyConfigured
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from .models import DataTable
import random
from django.db.models import Max, Min
import pickle
from .utils import push_file

from rest_framework.response import Response
from . import serializers
from django.shortcuts import render
from datetime import datetime


# Create your views here.
class AuthViewSet(viewsets.GenericViewSet):
    permission_classes = [AllowAny, ]
    serializer_class = serializers.EmptySerializer
    serializer_classes = {
        'enter': serializers.DataEntrySerializer,
    }
    queryset = ''

    @action(methods=['POST', ], detail=False)
    def enter(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        entry = DataTable(char = data['char'])
        entry.save()

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")

        file_name = dt_string + str(entry)
        file = open(file_name, mode = 'w')
        file.write(str(data['data']))
        file.close()

        push_file(file_name)


        return Response(status=status.HTTP_204_NO_CONTENT)

    def get_serializer_class(self):
        if not isinstance(self.serializer_classes, dict):
            raise ImproperlyConfigured("serializer_classes should be a dict mapping.")

        if self.action in self.serializer_classes.keys():
            return self.serializer_classes[self.action]
        return super().get_serializer_class()
