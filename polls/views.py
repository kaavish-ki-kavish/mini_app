from django.shortcuts import render

from django.contrib.auth import get_user_model, logout
from django.core.exceptions import ImproperlyConfigured
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from django.db.models import Max, Min
import os, datetime, random, copy

from .feature_extractor import  get_mahalanobis_distance

from rest_framework.response import Response
from . import serializers
from .utils import get_whole_stroke, get_feature_vector, feature_scorer, perfect_scorer, get_drawing_score_cnn

from django.http import JsonResponse
from .classifier import RandomForestClassifier
from .feature_extractor import hbr_feature_extract, scale_strokes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2 as cv


# Create your views here.
class AuthViewSet(viewsets.GenericViewSet):
    permission_classes = [AllowAny, ]
    serializer_class = serializers.EmptySerializer
    serializer_classes = {
        'enter': serializers.DataEntrySerializer,
        'get_score': serializers.DataEntrySerializerTf,
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

    @action(methods=['POST'], detail=False)
    def get_score(self, request):

        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        img = np.array(data['img'])
        img = img.astype(np.uint8)
        scores = []
        char = data['char']
        whole_x = data['whole_x']
        whole_y = data['whole_y']
        penup = data['pen_up']
        msg = 'Successful'

        if data['exercise'] == 0: #drawing
            categories = ['circle', 'triangle', 'bird', 'square', 'axe', 'airplane', 'apple', 'banana', 'arm', 'car']
            if char in categories:
                label = categories.index(char)
                scores.append(get_mahalanobis_distance(whole_x, whole_y, penup, label))
                scores.append(get_drawing_score_cnn(whole_x, whole_y, penup, label))
                print(scores)

            else:
                scores = [-1]

        elif data['exercise'] == 1:  # urdu letters
            print('here0')
            p_features, s_features = get_feature_vector(char)
            print('here1')
            scores.append(feature_scorer(img, p_features, s_features, verbose=1))
            print('here2')
            scores.append(perfect_scorer(whole_x, whole_y, penup, char))

        response = {
            'scores': scores,
        }

        return Response(response)
