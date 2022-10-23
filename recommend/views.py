from django.shortcuts import render, redirect
import pandas as pd
from . import analysis
from django.contrib import messages
# Create your views here.


def index(request):
    id = request.user.id
    print(request.user.id)  # 로그인 된 유저 아이디 정보

    if analysis.isin_data(id) == False:
        check_data = 'false'
    else:
        check_data = 'true'
    print(check_data)

    return render(request, 'recommend/index.html', {'check_data': check_data})


def choice_genres(request):
    if request.POST:
        list_item = request.POST.getlist('test_list')
        print('선택된 장르 : ', list_item)
        print('선택된 장르 갯수 : ', len(list_item))

        uid = request.user.id

        # 요청한 장르가 2개 미만 5개 이상일 경우
        if len(list_item) < 2 or len(list_item) > 3:
            messages.info(request, '2개 이상 선택해주세요.')
            # confirm('2개 이상 선택해주세요.')
        else:
            df = analysis.read_dataset()

            global recommend    # 전역 변수 선언
            # recommend = analysis.first_recommend(df, list_item, uid)
            recommend = analysis.initial_question(df, list_item, uid)
            print(recommend)
            return redirect('choice_games')

    return render(request, 'recommend/choice_genres.html')


def choice_games(request):
    # 리스트 출력
    # print(recommend)
    if request.POST:
        list_star = request.POST.getlist('list_stars')
        list_star = list(map(int, list_star))
        print(list_star)

        # 모든 값이 0인지 확인
        zero_cnt = 0
        for star in list_star:
            # print(star)
            if star != 0:
                break
            else:
                zero_cnt += 1

        if zero_cnt != 10:
            # 게임 데이터 불러오기
            games = analysis.read_dataset()

            # user id 가져오기
            id = request.user.id
            if id == None:    # 비로그인 시 0으로 선택
                id = 0

            ratings = analysis.make_ratings(id, recommend, games, list_star)

            data_folds = analysis.second_train_data()

            global result_games
            temp_games = analysis.second_game_recommend(
                data_folds, ratings, games, id)

            result_games = []
            for data in temp_games:
                temp_tags = data[3].split('/')

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
                    [data[0], data[1], data[2], game_tags, data[4], data[5]])

            print(result_games)

            # 추천 결과 저장하기
            analysis.recommend_store(result_games, id)

            return redirect('choice_result')

    return render(request, 'recommend/choice_games.html', {'recommend': recommend})


def choice_result(request):
    # 인기있는 태그 검색

    return render(request, 'recommend/choice_result.html', {'recommend': result_games})
