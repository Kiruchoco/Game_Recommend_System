from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'analysis'

urlpatterns = [
    # 평가하기(로그인 후 사용 가능)
    path('evaluation', views.evaluation, name='evaluation'),
    # 성향분석(로그인 후 사용 가능)
    path('tendency', views.tendency, name='tendency'),
    path('user_recommend', views.user_recommend,
         name='user_recommend'),  # 게임추천(로그인 후 사용 가능)
    # 자신이 추천한 게임 보기(로그인 후 사용가능)
    path('show_games', views.show_games, name='show_games'),
]
