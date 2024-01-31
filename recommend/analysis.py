import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.dataset import DatasetAutoFolds
from surprise.dataset import Reader, Dataset
from surprise import SVD
import random
import sqlite3


def isin_data(id):
    con = sqlite3.connect('db.sqlite3')                       # 데이터 베이스 연결
    ratings = pd.read_sql("SELECT * FROM game_rating", con)    # 플레이 데이터 불러오기
    con.close()  # 데이터베이스 종료

    target_ratings = ratings[ratings['userid'] == id]
    if len(target_ratings) == 0:
        return False
    else:
        return True


# 게임 정보 데이터 셋 불러오기
def read_dataset():
    # CSV 파일에서 불러오기
    # games=pd.read_csv("data/game_info.csv",encoding='latin_1')

    # 데이터베이스에서 불러오기
    con = sqlite3.connect('db.sqlite3')
    games = pd.read_sql("SELECT * FROM game_info", con)
    con.close()
    games = games.dropna()  # 결측치 제거

    # 데이터 형변환
    # games=games.astype({'appid':int})

    return games


def initial_question(games, choice_genre, id):
    # 로그인 하였을 경우
    if id != None:
        con = sqlite3.connect('db.sqlite3')      # 데이터 베이스 연결
        ratings = pd.read_sql("SELECT * FROM game_rating",
                              con)    # 플레이 데이터 불러오기
        con.close()  # 데이터베이스 종료

        # 내가 평가한 게임들의 ID 추출
        my_ratings = ratings[ratings['userid'] == id]['appid'].tolist()

        # 내가 평가했던 게임들을 제거함
        for data in my_ratings:
            games = games[games['appid'] != data]

    # 선택된 장르를 포함한 게임만 추출함
    for genre in choice_genre:
        games = games[games['genre'].str.contains(genre)]

    # 인기 순으로 정렬
    games = games.sort_values('reviews_cnt', ascending=False)

    # 데이터 프레임 인덱스 초기화
    games = games.reset_index(drop=True)

    recommend = []
    # 인기 순으로 10개 선정
    for idx in range(10):
        game_title = games.loc[idx, 'name']
        game_id = games.loc[idx, 'appid']
        tags = games.loc[idx, 'popular_tags']
        tags = list(tags.split(','))

        # 태그 5개 추출
        tag_cnt = 0
        game_tags = ''
        for tag in tags:
            if tag_cnt == 0:
                game_tags = tag
            else:
                game_tags += ' / ' + tag

            if tag_cnt == 5:
                break
            else:
                tag_cnt += 1

        recommend.append([game_title, game_id, game_tags])

    return recommend


# 코사인 유사도를 이용한 게임 추천
def first_recommend(games, choice_genre, userID):
    # 로그인 한 계정일 경우 처리한 이미 평가한 게임 삭제
    if userID != None:
        con = sqlite3.connect('db.sqlite3')      # 데이터 베이스 연결
        ratings = pd.read_sql("SELECT * FROM game_rating",
                              con)    # 플레이 데이터 불러오기
        con.close()  # 데이터베이스 종료

        # 내가 평가한 게임들의 ID 추출
        my_ratings = ratings[ratings['userid'] == userID]['appid'].tolist()

        # 내가 평가했던 게임들을 제거함
        for data in my_ratings:
            games = games[games['appid'] != data]

    # 선택된 장르를 포함한 게임만 추출함
    for genre in choice_genre:
        games = games[games['genre'].str.contains(genre)]

    # 인기 순으로 정렬
    games = games.sort_values('reviews_cnt', ascending=False)

    # 기준 게임 선택하기
    max_len = games.shape[0]  # 데이터 프레임 크기
    # target_ramdom = random.randrange(1, max_len)   # 랜덤으로 범위 내 게임 선택
    # print(target_ramdom)

    target_ramdom = 0
    standard_game = games.iloc[target_ramdom]['name']  # 기준 게임 이름 검색
    print('기준 게임 : ', standard_game)

    # 데이터 프레임 인덱스 초기화
    games = games.reset_index(drop=True)

    # 게임명과 태그를 결합한 열 생성
    games['important_features'] = get_important_features(games)  # 게임 장르와 이름 결합
    print('total shape : ', games.shape)

    # 행렬 변환(벡터화)
    cosine_matrix = CountVectorizer().fit_transform(
        games['important_features'])
    print("shape : ", cosine_matrix.shape)

    # 추천 기준이 되는 게임 인덱스 값 가져오기
    index_id = games[games.name == standard_game].index[0]

    # 변환한 행렬을 통해 코사인 유사도 측정
    cosine_similar = cosine_similarity(cosine_matrix, cosine_matrix)
    print('consine_similar : ', cosine_similar.shape)

    # 인덱스 값을 기준으로 정렬화
    scores = list(enumerate(cosine_similar[int(index_id)]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    # print(sorted_scores)

    # 기준 게임과 유사한 게임 10개 추출(코사인 유사도를 통한 추천)
    cnt = 0
    print('***', standard_game, '에 대한 10개의 유사한 게임***')
    recommend = []

    for item in sorted_scores:
        # print(item)
        game_title = games.iloc[item[0]]['name']
        print(game_title)
        # recommend.append(game_title)        # 추후 추천을 위한 추천 데이터 저장
        game_id = games.iloc[item[0]]['appid']
        # recommend.append([game_title, game_id])        # 추후 추천을 위한 추천 데이터 저장

        """
        # 간단한 설명 가져오기 ('https://store.steampowered.com/api/appdetails?appids=218620&l=korean')
        game_url = 'https://store.steampowered.com/api/appdetails?appids=' + \
            str(game_id)+'&l=korean'
        print(game_url)

        game_data = pd.read_json(game_url)

        if game_data[game_id]['success'] != 'false':
            game_descript = game_data[game_id]['data']['short_description']
            # print(game_data[game_id]['data'].keys())
        elif game_data[game_id]['success'] == 'false':
            game_descript = ''    # 해당 값이 없을 경우

        recommend.append([game_title, game_id, game_descript]
                         )        # 추후 추천을 위한 추천 데이터 저장
        """
        # 태그 정보 가져오기
        tags = list(games.iloc[item[0]]['popular_tags'].split(','))
        # print(tags)
        tag_cnt = 0
        game_tags = ''
        for tag in tags:
            if tag_cnt == 0:
                game_tags = tag
            else:
                game_tags += ' / ' + tag

            if tag_cnt == 5:
                break
            else:
                tag_cnt += 1

        recommend.append([game_title, game_id, game_tags]
                         )        # 추후 추천을 위한 추천 데이터 저장

        # 콘솔 출력
        print(cnt+1, '번째 추천할 게임')
        print('appid : ', game_id)
        print('title : ', game_title)
        print('tags : ', game_tags)
        print('Similarity : ', item[1])
        # print('Short Description : ', game_descript)

        cnt = cnt+1
        if cnt > 9:
            break

    return recommend


# 게임 이름과 태그를 결합한 열 생성
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['name'][i]+' '+data['popular_tags'][i])

    return important_features


# Second Recommendations
def make_ratings(id, recommend, games, list_star, max_len=10):
    # user_id 설정
    user_id = [id]*max_len

    # 게임 이름 입력
    game_name = []
    for i in range(len(recommend)):
        game_name.append(recommend[i][0])
    print(game_name)

    # user_data 생성(유저 아이디, 게임 이름, 별점)
    user_data = pd.DataFrame(
        {'userid': user_id, 'name': game_name, 'stars': list_star})
    # print(user_data)

    user_data = pd.merge(user_data, games, on='name')
    user_data = user_data[['userid', 'appid', 'stars']]

    # steam_raw = pd.read_csv("data/game_rating.csv", header=None, names=["userid", "appid", "stars"])
    # 게임 별점 정보 불러오기
    con = sqlite3.connect('db.sqlite3')
    steam_raw = pd.read_sql("SELECT * FROM game_rating", con)

    # print(steam_raw.columns)

    steam_raw = pd.concat([steam_raw, user_data])  # 값 추가
    # print(steam_raw.dtypes)

    # 값 저장하기
    steam_raw = steam_raw[steam_raw['stars'] != 0]
    if id != 0:  # 로그인일 경우에만 저장함
        steam_raw.to_sql("game_rating", con, if_exists="replace",
                         index=False)     # 데이터베이스 저장

    steam_raw.to_csv("data/game_ratings.csv", index=False,
                     header=None)   # 다음 추천을 위한 임시 파일 생성
    con.close()
    return steam_raw


def second_train_data():
    # 저장했던 데이터 불러오기
    reader = Reader(line_format='user item rating',
                    sep=',', rating_scale=(0.5, 5))

    # con = sqlite3.connect('db.sqlite3')
    # ratings = pd.read_sql("SELECT * FROM game_rating", con)
    # con.close()
    # data_folds = Dataset.load_from_df(ratings, reader=reader)

    data_folds = DatasetAutoFolds(
        ratings_file='data/game_ratings.csv', reader=reader)
    # ratings = pd.read_csv('data/game_ratings.csv',sep=',',names=['userID','appid','stars'])

    return data_folds


# 플레이한 게임과 플레이 하지 않은 게임 분류
def get_unplay_surprise(ratings, games, userID):
    seen_games = ratings[ratings['userid'] == userID]['appid'].tolist()

    total_games = games['appid'].tolist()

    unseen_games = [game for game in total_games if game not in seen_games]
    print(f'특정 {userID}번 유저가 플레이 한 게임 수: {len(seen_games)}\n')
    print(f'추천할 게임 개수: {len(unseen_games)}\n')
    print(f'전체 게임 수 : {len(total_games)}')

    return unseen_games


# 알고리즘 적용 후 게임 추천
def recommend_game_by_surprise(algo, userID, unseen_games, games, top_n=10):
    predictions = [algo.predict(str(userID), str(appid))
                   for appid in unseen_games]

    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions = predictions[:top_n]

    # 게임 id와 예측 점수 배열화
    top_game_ids = [int(pred.iid) for pred in top_predictions]
    top_game_ratings = [round(pred.est, 2) for pred in top_predictions]

    # 게임 목록에서 제목과 장르 찾아서 저장함
    top_game_preds = []
    for idx in range(len(top_predictions)):
        top_game_title = games[games['appid'] ==
                               top_game_ids[idx]]['name'].values

        top_game_genres = games[games['appid']
                                == top_game_ids[idx]]['genre'].values
        top_game_genres = top_game_genres[0].split(",")
        # print(top_game_genres)

        top_game_tags = games[games['appid'] ==
                              top_game_ids[idx]]['popular_tags'].values
        top_game_tags = list(top_game_tags)
        top_game_tags = top_game_tags[0].split(",")
        print(top_game_tags)

        tag_cnt = 0
        game_tags = ''
        for tag in top_game_tags:
            if tag_cnt == 0:
                game_tags = tag
            else:
                game_tags += ' / ' + tag

            if tag_cnt == 5:
                break
            else:
                tag_cnt += 1

        top_game_developer = games[games['appid'] ==
                                   top_game_ids[idx]]['developer'].values

        # print(top_game_ids[idx], top_game_ratings[idx], top_game_title, top_game_genres)
        top_game_preds.append(
            [top_game_ids[idx], top_game_ratings[idx], top_game_title[0], game_tags, top_game_genres[0], top_game_developer[0]])

    return top_game_preds


def second_game_recommend(data_folds, ratings, games, userid):
    # 저장했던 데이터 불러오기
    # reader = Reader(line_format='userid appid stars')

    # data_folds = DatasetAutoFolds(df=ratings, reader=reader)

    # 전부 훈련 데이터로 사용함
    trainset = data_folds.build_full_trainset()
    algo = SVD(n_factors=5, n_epochs=30, lr_all=0.005, reg_all=0.1)
    algo.fit(trainset)

    unseen_lst = get_unplay_surprise(ratings, games, userid)
    top_game_preds = recommend_game_by_surprise(
        algo, userid, unseen_lst, games, top_n=10)
    # print(top_game_preds)

    # 추천 게임 리스트
    print()
    print('#'*8, 'Top-20 추천게임 리스트', '#'*8)
    game_cnt = 1
    for top_game in top_game_preds:
        print('* ', game_cnt, '번째 추천 게임')
        print('* 추천 게임 번호: ', top_game[0])
        print('* 해당 게임의 예측평점: ', top_game[1])
        print('* 추천 게임 이름: ', top_game[2])
        print('* 해당 게임의 태그: ', top_game[3])
        print('* 해당 게임의 장르: ', top_game[4])
        print('* 해당 게임의 개발사: ', top_game[5])

        print()
        game_cnt = game_cnt+1

    return top_game_preds


# 추천 데이터 저장
def recommend_store(result_games, id):
    # 기존 데이터 삭제
    con = sqlite3.connect('db.sqlite3')                  # 데이터 베이스 연결
    cur = con.cursor()    # 커서 등록
    sql_query = "DELETE FROM game_recommend WHERE userid==" + \
        str(id)  # 해당 ID 추천 기록 제거하기
    cur.execute(sql_query)      # 추천 데이터 삭제
    con.commit()                # 데이터 베이스 반영

    for result_game in result_games:
        sql_query = "INSERT INTO game_recommend(userid, appid, stars, name, popular_tags, genre, developer) values (?, ?, ?, ?, ?, ?, ?)"
        cur.execute(sql_query, (id, result_game[0], result_game[1],
                    result_game[2], result_game[3], result_game[4], result_game[5]))
        # print(result_game)
    con.commit()                # 데이터 베이스 반영

    con.close()  # 데이터베이스 종료
