from django.shortcuts import render, redirect
from . import users
import sqlite3
import pandas as pd

# Create your views here.


def tendency(request):
    id = request.user.id

    # 데이터가 없는 경우
    if users.isin_data(id) == False:
        return redirect('index')
        messages.warning(request, "데이터가 없습니다.")
    # 데이터가 있는 경우
    else:
        list_genre = users.calc_genres(id)
        list_tags = users.calc_tags(id)
        list_developer = users.calc_developer(id)

        # 출력할 데이터 list 만들기
        list_recommend = []
        list_recommend.append(list_genre[0])
        list_recommend.append(list_genre[1])
        list_recommend.append(list_tags[0])
        list_recommend.append(list_tags[1])
        list_recommend.append(list_developer[0])
        list_recommend.append(list_developer[1])
        # print(list_recommend)

        total_stars = users.calc_stars(id)
        # print(total_stars)

        return render(request, 'analysis/tendency.html', {"list_recommend": list_recommend, "list_stars": total_stars})


def evaluation(request):
    # 아이디 가져오기
    id = request.user.id

    # 데이터가 없는 경우
    if users.isin_data(id) == False:
        return redirect('index')
    else:
        global result_games
        con = sqlite3.connect('db.sqlite3')                       # 데이터 베이스 연결
        id = request.user.id
        sql_query = "SELECT appid, stars, name, popular_tags, genre FROM game_recommend WHERE appid==" + \
            str(id)+";"
        result_games = pd.read_sql(
            "SELECT appid, stars, name, popular_tags, genre FROM game_recommend WHERE userid=="+str(id), con)
        con.close()

        result_games = result_games.values.tolist()
        # print(result_games)

        if request.POST:
            list_star = request.POST.getlist('list_stars')
            list_star = list(map(int, list_star))
            # print(list_star)

            # 게임 데이터 불러오기
            games = users.read_dataset()

            # 데이터 통합
            ratings = users.make_ratings(id, result_games, games, list_star)

            # 게임 추천(사용자 기반)

            # 훈련 데이터 가져오기
            ratings = users.temp_ratings()
            data_folds = users.second_train_data()

            result_games = users.second_game_recommend(
                data_folds, ratings, games, id)

            # print(result_games)

            # 데이터 베이스 저장
            users.recommend_store(result_games, id)

            return redirect('index')

    return render(request, 'analysis/evaluation.html', {'recommend': result_games})


def user_recommend(request):
    # 아이디 가져오기
    id = request.user.id

    # 데이터가 없는 경우
    if users.isin_data(id) == False:
        return redirect('index')
    else:
        # 게임 정보 가져오기
        games = users.read_dataset()

        # 훈련 데이터 가져오기
        ratings = users.temp_ratings()
        data_folds = users.second_train_data()

        # 게임 추천에서 태그 가져오기
        con = sqlite3.connect('db.sqlite3')         # 데이터 베이스 연결
        temp_games = pd.read_sql(
            "SELECT appid, stars, popular_tags, genre, developer FROM game_recommend WHERE userid=="+str(id), con)
        con.close()
        # print(result_games)

        temp_games = temp_games.values.tolist()
        result_games = []

        # 태그 3개 추출
        for data in temp_games:
            temp_tags = data[2].split('/')

            tag_cnt = 0
            game_tags = ''
            for tag in temp_tags:
                if tag_cnt == 0:
                    game_tags = tag
                else:
                    game_tags += ' / ' + tag

                if tag_cnt == 2:
                    break
                else:
                    tag_cnt += 1

            result_games.append(
                [data[0], data[1], game_tags, data[3], data[4]])

        # print('추천 게임 : ', result_games)

        # 가장 선호하는 게임에 대한 유사한 게임 검색
        global result_recommend
        result_recommend = users.cosine_recommend(games, ratings, id)

        # print(result_recommend)

        # 가장 선호하는 태그에 대한 인기 게임 검색
        popular_tag, top_tags = users.popular_tag_game(games, ratings, id)
        # print(popular_tag, top_tags)

        # 가장 선호하는 개발사에 대한 인기 게임 검색
        popular_developer, top_developer = users.popular_company_game(
            games, ratings, id)

    return render(request, 'analysis/user_recommend.html', {'recommend': result_games, 'cosine': result_recommend, 'top_tags': top_tags, 'popular_tags': popular_tag, 'top_developer': top_developer, 'popular_developer': popular_developer})


def show_games(request):
    # 아이디 불러오기
    id = request.user.id

    # 데이터가 없는 경우 메인 화면으로
    if users.isin_data(id) == False:
        return redirect('index')

    # 게임 데이터셋 가져오기
    games = users.read_dataset()

    # 나의 게임 데이터 셋 산출
    mygames = users.my_all_games(id, games)
    total = len(mygames)

    return render(request, 'analysis/show_games.html', {'mygames': mygames, 'total': total})
