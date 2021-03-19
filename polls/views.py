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

from .models import ChildProfile, Characters, Session, History, ObjectWord, ColoringExercise, DrawingExercise

from rest_framework.response import Response
from . import serializers
from .utils import get_whole_stroke

from django.http import JsonResponse
from .classifier import RandomForestClassifier
from .feature_extractor import hbr_feature_extract, scale_strokes
from .urduCNN import UrduCnnScorer
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision
import sys
from torchvision import datasets, transforms
from torch.utils import data
import torch.nn as nn
import requests


# Create your views here.
class AuthViewSet(viewsets.GenericViewSet):
    permission_classes = [AllowAny, ]
    serializer_class = serializers.EmptySerializer
    serializer_classes = {
        'enter': serializers.DataEntrySerializer,
        'get_score': serializers.DataEntrySerializer,
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
        scores = []
        whole_x, whole_y, penup = get_whole_stroke(data['data'])
        char = data['char']

        url = 'http://aanganmodelstf.herokuapp.com/models/get_score'

        msg = 'Successful'

        if data['exercise'] == 0:  # drawing
            tf_data = {
                'exercise': 0,
                'char': char,
                'img': [[0, 0]],
                'whole_x': whole_x,
                'whole_y': whole_y,
                'pen_up': list(penup)
            }
            response = requests.post(url, json=tf_data)
            print(response)
            print(type(response))
            sys.stdout.flush()
            print(response.json()['scores'])
            sys.stdout.flush()
            a = response.json()['scores'][0]  # feature_scorer(img, p_features,s_features, verbose= 1)
            b = response.json()['scores'][1]  # perfect_scorer(whole_x, whole_y, penup, char)
            scores.append(a)
            scores.append(b)



        elif data['exercise'] == 1:  # urdu letters
            scorer = UrduCnnScorer(whole_x, whole_y, penup)
            label = scorer.NUM2LABEL.index(char)
            img = scorer.preprocessing()
            print(img.shape)
            scores.append(scorer.test_img(img)[0, label])

            tf_data = {
                'exercise': 1,
                'char': char,
                'img': img.tolist(),
                'whole_x': whole_x,
                'whole_y': whole_y,
                'pen_up': list(penup)
            }

            response = requests.post(url, json=tf_data)
            print(response)
            print(type(response))
            sys.stdout.flush()
            print(response.json()['scores'])
            sys.stdout.flush()
            # p_features, s_features = get_feature_vector(char)
            a = response.json()['scores'][0]  # feature_scorer(img, p_features,s_features, verbose= 1)
            b = response.json()['scores'][1]  # perfect_scorer(whole_x, whole_y, penup, char)
            scores.append(a)
            scores.append(b)

        print(scores)
        response = {
            'message': msg,
            'prediction': np.mean(scores),
        }

        return Response(response)