from django.db import models

# Create your models here.
class Game_Info(models.Model):
    appID=models.IntegerField()
    url=models.CharField(max_length=1000)
    name=models.CharField(max_length=1000)
    desc_snippet=models.TextField()
    all_reviews=models.TextField()
    popular_tags=models.CharField(max_length=1000)
    genre=models.CharField(max_length=1000)


class Game_Rating(models.Model):
    userID=models.IntegerField()
    appID=models.IntegerField()
    stars=models.IntegerField()