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


def calc_genres(id):
    con = sqlite3.connect('db.sqlite3')                       # 데이터 베이스 연결
    ratings = pd.read_sql("SELECT * FROM game_rating", con)    # 플레이 데이터 불러오기
    games = pd.read_sql("SELECT * FROM game_info", con)        # 게임 데이터 불러오기
    games = games.dropna()  # 결측치 제거
    games = games.astype({'appid': int})
    con.close()  # 데이터베이스 종료

    target_genres = pd.merge(ratings, games, on='appid')
    # 해당 사용자 데이터를 제외한 나머지 제거
    target_genres = target_genres[target_genres['userid'] == id]

    target_genres = target_genres['genre'].str.split(',')
    target_genres = target_genres.str.get(0)  # 첫번째 장르만 남기고 모두 제거

    result_genres = target_genres.value_counts()
    result_genres = dict(result_genres)
    result_key = result_genres.keys()
    result_key = list(result_key)  # 장르 배열로 만들기
    result_key = '//'.join([str(i) for i in list(result_key)])  # 구분자 변경
    # print(result_key)
    result_count = result_genres.values()
    result_count = list(result_count)
    # print(result_count)

    result = []
    result.append(result_key)
    result.append(result_count)

    # print(result[0])
    # print(result[1])

    return result


def calc_tags(id):
    con = sqlite3.connect('db.sqlite3')                       # 데이터 베이스 연결
    ratings = pd.read_sql("SELECT * FROM game_rating", con)    # 플레이 데이터 불러오기
    games = pd.read_sql("SELECT * FROM game_info", con)        # 게임 데이터 불러오기
    games = games.dropna()  # 결측치 제거
    games = games.astype({'appid': int})
    con.close()  # 데이터베이스 종료

    # 게임 데이터와 플레이 데이터 조인
    target_tags = pd.merge(ratings, games, on='appid')

    # 해당 사용자 데이터를 제외한 나머지 제거
    target_tags = target_tags[target_tags['userid'] == id]

    target_tags = target_tags['popular_tags'].str.split(',')
    total_tags = []
    for tags in target_tags:
        for tag in tags:
            total_tags.append(tag)
    total_tags = pd.Series(total_tags)
    # target_tags = target_tags.str.get(0)  # 첫번째 태그만 남기고 모두 제거

    # 태그 배열 만들기
    result_tags = total_tags.value_counts()
    result_tags = dict(result_tags)

    result_key = result_tags.keys()
    result_key = list(result_key)
    # 태그 갯수가 9개가 넘어갈 경우

    # if len(result_key) >= 9:
    # result_key = result_key[0:9]
    # print('before : ', result_key)

    result_key = '//'.join([str(i) for i in list(result_key)])  # 구분자 변경
    # print('after : ', result_key)
    # print(result_key)
    result_count = result_tags.values()
    result_count = list(result_count)
    # 태그 갯수가 9개가 넘어갈 경우
    # if len(result_count) >= 9:
    # result_count = result_count[0:9]
    # print(result_count)

    # print(result_key)
    # print(result_count)

    result = []
    result.append(result_key)
    result.append(result_count)

    # print(result[0])
    # print(result[1])

    return result


def calc_stars(id):
    con = sqlite3.connect('db.sqlite3')                       # 데이터 베이스 연결
    ratings = pd.read_sql("SELECT * FROM game_rating", con)    # 플레이 데이터 불러오기
    con.close()  # 데이터베이스 종료

    target_stars = ratings[ratings['userid'] == id]    # 해당 사용자 데이터를 제외한 나머지 제거
    target_stars = target_stars['stars']

    star_list = [0, 0, 0, 0, 0]
    for data in target_stars:
        if data == 1:
            star_list[0] += 1
        elif data == 2:
            star_list[1] += 1
        elif data == 3:
            star_list[2] += 1
        elif data == 4:
            star_list[3] += 1
        elif data == 5:
            star_list[4] += 1

    return star_list


def calc_developer(id):
    con = sqlite3.connect('db.sqlite3')                       # 데이터 베이스 연결
    ratings = pd.read_sql("SELECT * FROM game_rating", con)    # 플레이 데이터 불러오기
    games = pd.read_sql("SELECT * FROM game_info", con)        # 게임 데이터 불러오기
    games = games.dropna()  # 결측치 제거
    games = games.astype({'appid': int})
    con.close()  # 데이터베이스 종료

    # 게임 데이터와 플레이 데이터 조인
    target_developer = pd.merge(ratings, games, on='appid')

    # 해당 사용자 데이터를 제외한 나머지 제거
    target_developer = target_developer[target_developer['userid'] == id]
    target_developer = target_developer['developer']

    # 태그 배열 만들기
    result_developer = target_developer.value_counts()
    result_developer = dict(result_developer)
    result_key = result_developer.keys()
    result_key = list(result_key)

    # 태그 갯수가 8개가 넘어갈 경우
    if len(result_key) >= 8:
        result_key = result_key[0:8]
    # print('before : ', result_key)

    result_key = '//'.join([str(i) for i in list(result_key)])  # 구분자 변경
    # print('after : ', result_key)
    # print(result_key)
    temp_count = result_developer.values()
    temp_count = list(temp_count)
    # print('총 갯수 : ', sum(temp_count))

    # 태그 갯수가 8개가 넘어갈 경우
    if len(temp_count) >= 8:
        result_count = temp_count[0:8]
        # 8개를 넘어갈 경우 나머지 추가
        remain_cnt = sum(temp_count[8:])
        result_count.append(remain_cnt)
        # print('나머지 갯수 : ', remain_cnt)
        result_key += '//etc'
    else:
        result_count = temp_count
    # print(result_key)
    # print(result_count)

    result = []
    result.append(result_key)
    result.append(result_count)

    # print(result[0])
    # print(result[1])

    return result


def make_ratings(id, recommend, games, list_star, max_len=10):
    # user_id 설정
    user_id = [id]*max_len

    game_name = []
    for i in range(max_len):
        game_name.append(recommend[i][2])

    print(game_name)

    # user_data 생성
    user_data = pd.DataFrame(
        {'userid': user_id, 'name': game_name, 'stars': list_star})
    # print(user_data)

    user_data = pd.merge(user_data, games, on='name')
    print(user_data)
    user_data = user_data[['userid', 'appid', 'stars']]

    # steam_raw = pd.read_csv("data/game_rating.csv", header=None, names=["userid", "appid", "stars"])
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
        print("data clear")
    steam_raw.to_csv("data/game_ratings.csv", index=False,
                     header=None)   # 다음 추천을 위한 임시 파일 생성
    con.close()
    return steam_raw


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


def temp_ratings():
    # steam_raw = pd.read_csv("data/game_rating.csv", header=None, names=["userid", "appid", "stars"])
    con = sqlite3.connect('db.sqlite3')

    steam_raw = pd.read_sql("SELECT * FROM game_rating", con)

    con.close()

    return steam_raw


def second_train_data():
    # 저장했던 데이터 불러오기
    reader = Reader(line_format='user item rating',
                    sep=',', rating_scale=(0.5, 5))
    data_folds = DatasetAutoFolds(
        ratings_file='data/game_ratings.csv', reader=reader)

    # ratings = pd.read_csv('data/game_ratings.csv',sep=',',names=['userID','appid','stars'])

    return data_folds


# 플레이하지 않은 게임 불러오기
def get_unplay_surprise(ratings, games, userID):
    seen_games = ratings[ratings['userid'] == userID]['appid'].tolist()

    total_games = games['appid'].tolist()

    unseen_games = [game for game in total_games if game not in seen_games]
    print(f'특정 {userID}번 유저가 플레이 한 게임 수: {len(seen_games)}\n')
    print(f'추천할 게임 개수: {len(unseen_games)}\n')
    print(f'전체 게임 수 : {len(total_games)}')

    return unseen_games


# 알고리즘 적용 후 게임 추천
def recommend_game_by_surprise(algo, userID, unseen_games, games, top_n=20):
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
        # 태그 정보 가져오기
        top_game_tags = games[games['appid'] ==
                              top_game_ids[idx]]['popular_tags'].values
        top_game_tags = top_game_tags[0].split(",")
        # print(tags)
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
    # 전부 훈련 데이터로 사용함
    trainset = data_folds.build_full_trainset()
    algo = SVD(n_factors=5, n_epochs=30, lr_all=0.005, reg_all=0.1)
    algo.fit(trainset)

    unseen_lst = get_unplay_surprise(ratings, games, userid)
    top_game_preds = recommend_game_by_surprise(
        algo, userid, unseen_lst, games, top_n=10)
    # print(top_game_preds)
    print()
    print('#'*8, 'Top-10 추천게임 리스트', '#'*8)
    game_cnt = 1
    for top_game in top_game_preds:
        print('* ', game_cnt, '번째 추천 게임')
        print('* 추천 게임 번호: ', top_game[0])
        print('* 해당 게임의 예측평점: ', top_game[1])
        print('* 추천 게임 이름: ', top_game[2])
        print('* 해당 게임의 태그: ', top_game[3])
        print('* 해당 게임의 장르: ', top_game[4])
        print('* 해당 게임의 제작사: ', top_game[5])

        print()
        game_cnt = game_cnt+1

    return top_game_preds


def cosine_recommend(games, ratings, uid):
    # 해당 유저의 플레이 게임 불러오기
    user_ratings = ratings[ratings['userid'] == uid]
    user_ratings = user_ratings.sort_values(
        'stars', ascending=False)  # 별점을 기준으로 정렬

    # 가장 선호하는 게임 1개 선택
    standard_id = user_ratings.iloc[0]
    standard_id = standard_id['appid']
    print(standard_id)

    # 해당 게임의 장르 추출
    standard_game = games[games['appid'] == standard_id]
    standard_genres = list(standard_game['genre'])
    standard_genres = standard_genres[0].split(',')

    # 장르가 1개인 경우
    if len(standard_genres) == 1:
        # 해당 장르가 포함된 게임만 추출
        games = games[games['genre'].str.contains(standard_genres[0])]
        print(standard_genres[0])
    # 장르가 2개 이상인 경우
    else:
        # 장르 2개 선택하여 해당 장르가 포함된 게임만 추출
        for genre in standard_genres[0:2]:
            games = games[games['genre'].str.contains(genre)]
            print(genre)

    # 데이터 프레임 인덱스 초기화
    games = games.reset_index(drop=True)

    # 게임명과 태그를 결합한 열 생성
    games['important_features'] = get_important_features(games)  # 게임 장르와 이름 결합
    print('total shape : ', games.shape)

    # 추천 기준이 되는 게임 인덱스 값 가져오기
    index_id = games[games.appid == standard_id].index[0]
    # print(index_id)

    # 행렬 변환(벡터화)
    cosine_matrix = CountVectorizer().fit_transform(
        games['important_features'])
    print("shape : ", cosine_matrix.shape)

    # 변환한 행렬을 통해 코사인 유사도 측정
    cosine_similar = cosine_similarity(cosine_matrix, cosine_matrix)
    # cosine_similar = cosine_similarity(target_matrix[0], cosine_matrix)
    print('consine_similar : ', cosine_similar.shape)

    scores = list(enumerate(cosine_similar[int(index_id)]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # 기준 게임과 유사한 게임 10개 추출(코사인 유사도를 통한 추천)
    cnt = 0
    print('***', standard_game, '에 대한 10개의 유사한 게임***')
    result_recommend = []

    for item in sorted_scores:
        game_title = games.iloc[item[0]]['name']
        print(game_title)
        game_id = games.iloc[item[0]]['appid']

        # 장르 정보 가져오기
        temp_genre = list(games.iloc[item[0]]['genre'].split(','))
        game_genres = temp_genre[0]

        # 태그 정보 가져오기
        temp_tags = list(games.iloc[item[0]]['popular_tags'].split(','))
        # print('태그 정보 : ', temp_tags)

        # print(tags)
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

        game_developer = games.iloc[item[0]]['developer']

        # 추후 추천을 위한 추천 데이터 저장
        result_recommend.append(
            [game_title, game_id, game_genres, game_tags, game_developer])

        # 콘솔 출력
        print(cnt+1, '번째 추천할 게임')
        print('appid : ', game_id)
        print('title : ', game_title)
        print('genre : ', game_genres)
        print('tags : ', game_tags)
        print('developers : ', game_developer)
        print('Similarity : ', item[1])
        # print('Short Description : ', game_descript)

        cnt = cnt+1
        if cnt > 9:
            break

    return result_recommend


# 게임 이름과 태그를 결합한 열 생성
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        important_features.append(data['name'][i]+' '+data['popular_tags'][i])

    return important_features


# 현재 사용자가 가지고 있는 모든 게임 산출
def my_all_games(id, games):
    # 데이터 가져오기
    con = sqlite3.connect('db.sqlite3')                       # 데이터 베이스 연결
    ratings = pd.read_sql("SELECT * FROM game_rating", con)    # 플레이 데이터 불러오기
    con.close()  # 데이터베이스 종료

    # 사용자 데이터 가져오기
    target_ratings = ratings[ratings['userid'] == id]
    print('사용자의 게임 갯수 : ', len(target_ratings))

    # 사용자의 게임 데이터 중 평점과 게임 아이디를 가져옴
    mygames = []
    my_id = target_ratings['appid'].to_list()
    my_star = target_ratings['stars'].to_list()

    for i in range(len(target_ratings)):
        # 별점 제작하기
        temp = int(my_star[i])
        stars = '☆'*(5-temp) + '★' * temp

        # 게임 태그 가져오기
        my_tags = games[games['appid'] == my_id[i]]['popular_tags'].values
        print(my_tags)
        # tag_split = my_tags['popular_tags'].str.split(',')
        # my_tags = list(my_tags['popular_tags'])
        # print(my_tags)
        my_tags = my_tags[0].split(',')
        print(my_tags)

        tag_cnt = 0
        game_tags = ''
        """
        for tag_cnt in range(3):
            if tag_cnt == 0:
                game_tags = tag_split.str.get(tag_cnt)
            else:
                game_tags += ' / ' + tag_split.str.get(tag_cnt)
        """

        for tag in my_tags:
            if tag_cnt == 0:
                game_tags = tag
            else:
                game_tags += ' / ' + tag

            if tag_cnt == 2:
                break
            else:
                tag_cnt += 1

        if i % 2 == 0:
            mygames.append([my_id[i], stars, 0, game_tags])
        else:
            mygames.append([my_id[i], stars, 1, game_tags])
        """
        if len(target_ratings) >= i+2:
            mygames.append([my_id[i], my_star[i], my_id[i+1], my_star[i+1]])
        else:
            mygames.append([my_id[i], my_star[i]])
        """
    print(mygames)

    return mygames


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


def popular_tag_game(games, ratings, id):
    top_tags = calc_tags(id)

    # 가장 인기있는 태그 추출
    top_tags = top_tags[0]
    top_tags = list(top_tags.split('//'))
    top_tags = top_tags[0]

    print('가장 인기있는 태그', top_tags)

    # 이미 플레이 한 게임 제거
    # 내가 평가한 게임들의 ID 추출
    my_ratings = ratings[ratings['userid'] == id]['appid'].tolist()

    # 내가 평가했던 게임들을 제거함
    for data in my_ratings:
        games = games[games['appid'] != data]

    # 해당 태그가 들어있는 게임 가져오기
    games = games[games['popular_tags'].str.contains(top_tags)]

    # 인기순으로 정렬함
    popular_tags = games.sort_values(
        'reviews_cnt', ascending=False)  # 리뷰 갯수를 기준으로 정렬

    # 가장 인기있는 게임 5개 추출
    return popular_tags.head(5), top_tags


def popular_company_game(games, ratings, id):
    top_company = calc_developer(id)

    # 가장 인기있는 개발사 추출
    top_company = top_company[0]
    top_company = list(top_company.split('//'))
    top_company = top_company[0]

    print('가장 인기있는 개발사', top_company)

    # 내가 평가한 게임들의 ID 추출
    my_ratings = ratings[ratings['userid'] == id]['appid'].tolist()

    # 내가 평가했던 게임들을 제거함
    for data in my_ratings:
        games = games[games['appid'] != data]

    # 해당 게임의 개발사 찾기
    games = games[games['developer'].str.contains(top_company)]

    # 인기순으로 정렬함
    popular_developer = games.sort_values(
        'reviews_cnt', ascending=False)  # 리뷰 갯수를 기준으로 정렬

    # 인덱스 초기화
    games = games.reset_index(drop=True)

    # 가장 인기있는 게임 5개 추출
    result_developer = []
    for idx in range(len(popular_developer)):
        game_title = games.loc[idx, 'name']
        game_id = games.loc[idx, 'appid']
        game_tags = games.loc[idx, 'tags_brief']
        game_developer = games.loc[idx, 'developer']

        # 색 변환을 위한 값 넣기
        if idx % 2 == 0:
            result_developer.append(
                [game_title, game_id, game_tags, game_developer, 0])
        else:
            result_developer.append(
                [game_title, game_id, game_tags, game_developer, 1])

        # 5개가 선정되었을 경우 종료
        if idx == 4:
            break

    print(result_developer)

    return result_developer, top_company
