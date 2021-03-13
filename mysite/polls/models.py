from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from datetime import datetime

# Create your models here.

class DataTable(models.Model):
    id = models.AutoField(primary_key = True)
    creation_date = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    char = models.CharField(max_length= 255)

    def __str__(self):
        return f'{self.id}_{self.char}'



