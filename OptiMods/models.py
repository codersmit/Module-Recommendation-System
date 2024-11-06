from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser): # OptiMods custom User model
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    preferences = models.TextField(blank=True)
    career_aspirations = models.TextField(blank=True)

    def __str__(self):
        return self.email # return email as it's an easy way to identify each user cos it's unique to the user

