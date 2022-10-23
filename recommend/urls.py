from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),                                  # 메인페이지
    path('index',views.index,name='index'),                             # 메인페이지
    path('choice_genres',views.choice_genres,name='choice_genres'),     # 장르 선택페이지
    path('choice_games',views.choice_games,name='choice_games'),        # 장르 기반 추천 후 선택 페이지
    path('choice_result',views.choice_result,name='choice_result'),     # 유사 사용자 기반 추천 페이지
]