from django.shortcuts import render, redirect
import pandas as pd
from . import analysis

# Create your views hee.
def index(request):
    print(request.user.id)  # 로그인 된 유저 아이디 정본
    return render(request,'recommend/index.html')

def testpage(request):
    if request.POST:
        list_item=request.POST.getlist('test_list')
        
        genre=list_item[0]
        print(genre)

        df=analysis.read_dataset()
        standard_game=analysis.find_standards(df,genre)   # 기준 게임 선정
        
        global recommend    # 전역 변수 선언
        recommend=analysis.first_recommend(df,standard_game)
        return redirect('first')
    
    return render(request,'recommend/testpage.html')

def first(request):
    # 리스트 출력
    # print(recommend)
    if request.POST:
        """
        list_star=request.POST.getlist('flexRadioDefault')
        #genre=list_item[0]
        print(list_star)
        
        print(list_star)
        """
        gameno1=request.POST.getlist('flexRadioDefault')
        gameno2=request.POST.getlist('flexRadioDefault2')
        gameno3=request.POST.getlist('flexRadioDefault3')
        gameno4=request.POST.getlist('flexRadioDefault4')
        gameno5=request.POST.getlist('flexRadioDefault5')
        gameno6=request.POST.getlist('flexRadioDefault6')
        gameno7=request.POST.getlist('flexRadioDefault7')
        gameno8=request.POST.getlist('flexRadioDefault8')
        gameno9=request.POST.getlist('flexRadioDefault9')
        gameno10=request.POST.getlist('flexRadioDefault10')
        list_star=gameno1+gameno2+gameno3+gameno4+gameno5+gameno6+gameno7+gameno8+gameno9+gameno10
        list_star=list(map(int,list_star))
        print(list_star)
        
        # 별점 입력받기
        #list_star=list(map(int,input("개임 10개에 대한 별점 입력 : ").split()))
        #list_star=[5,3,2,5,0,0,3,4,5,5]     # 10개의 추천 항목에 대한 별점(0~5점까지 줄 수 있음)
        
        #list_star=analysis.stars_to_hours(list_star) # 별점 시간으로 임시 환산
        # 게임 데이터 불러오기  
        games=analysis.read_dataset()
        #games=games[['appid','name']]
        
        # user id 가져오기
        id=request.user.id
        if id==None:    # 비로그인 시 0으로 선택
            id=0

        ratings=analysis.make_ratings(id, recommend, games, list_star)
        #ratings=analysis.second_preprocessing(ratings, games)
        #ratings=pd.read_csv("data/game_rating.csv")
        #print(ratings.shape)

        data_folds=analysis.second_train_data()

        global result_games
        result_games=analysis.second_game_recommend(data_folds, ratings, games)

        return redirect('second')

    return render(request,'recommend/first.html',{'recommend': recommend})

def second(request):
    return render(request,'recommend/second.html',{'recommend': result_games})

