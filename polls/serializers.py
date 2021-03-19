from django.contrib.auth import get_user_model, password_validation
from rest_framework.authtoken.models import Token
from rest_framework import serializers

class DataEntrySerializer(serializers.Serializer):
    char = serializers.CharField(max_length= 255, required= True)
    data = serializers.ListField(
        child=serializers.ListField(
            child = serializers.ListField(
            child= serializers.IntegerField())))

class EmptySerializer(serializers.Serializer):
    pass

class DataEntrySerializerTf(serializers.Serializer):
    char = serializers.CharField(max_length= 255, required= True)
    img = serializers.ListField(
        child=serializers.ListField(
            child= serializers.IntegerField())
    )
    whole_x = serializers.ListField(
                child= serializers.IntegerField()
            )
    whole_y = serializers.ListField(
                child= serializers.IntegerField()
            )
    pen_up = serializers.ListField(
                child= serializers.IntegerField()
            )
    exercise = serializers.IntegerField()
