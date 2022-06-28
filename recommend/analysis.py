import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise.dataset import DatasetAutoFolds
from surprise.dataset import Reader, Dataset
from surprise import SVD
import random
import sqlite3

def read_dataset():
    # 데이터 셋 불러오기
    # games=pd.read_csv("data/game_info.csv",encoding='latin_1')
    # 데이터베이스에서 불러오기
    con=sqlite3.connect('db.sqlite3')
    games=pd.read_sql("SELECT * FROM game_info",con)
    con.close()
    games=games.dropna() # 결측치 제거
    games=games.astype({'appid':int})

    return games


def find_standards(df, choice_genre):
    # 선택한 장르를 포함하고 있는 게임 검색
    #choice_genre=genre_list[choices]
    df_rs=df[df['genre'].str.contains(choice_genre)]

    # 해당 장르 조건을 충족하는 게임 중 랜덤으로 기준 게임 채택
    max_len=df_rs.shape[0]
    target_random=random.randrange(1,max_len)
    
    standard_games=df_rs.iloc[target_random]['name']
    print(standard_games)

    return standard_games


def first_recommend(df, standard_game):
    # 게임 추천 알고리즘
    df=df.reset_index(drop=True)

    # 유사도를 산출할 지표(장르, 게임 이름)
    columns=['genre','name']
    # print(df[columns].head(10))

    # 결측값 찾기
    df[columns].isnull().values.any()
    
    df['important_features']=get_important_features(df) # 게임 장르와 이름 결합

    # 행렬 변환
    cm=CountVectorizer().fit_transform(df['important_features'])
    # print(cm)

    # 변환한 행렬을 통해 코사인 유사도 측정
    cs=cosine_similarity(cm)
    # print(cs)

    # 추천 기준이 되는 게임 인덱스 값 가져오기
    app_id=df[df.name==standard_game]['appid'].values[0]
    #print(int(app_id))

    # 인덱스 값을 기준으로 정렬화
    scores=list(enumerate(cs[int(app_id)]))
    sorted_scores = sorted(scores,key = lambda x:x[1],reverse = True)
    sorted_scores = sorted_scores[1:]
    #print(sorted_scores)

    # 초기 추천할 10개의 게임 선정
    j = 0
    print ('10 recomended game for',standard_game,'are:\n')
    recommend=[]

    for item in sorted_scores:
        print(item)
        game_title = (df[df.appid== item[0]]['name'].values[0])
        print(game_title)
        recommend.append(game_title)        # 추후 추천을 위한 추천 데이터 저장
        urls=(df[df.appid == item[0]]['url'].values[0])
        print(j+1,game_title)
        print('Similarity : ',end='')
        print(item[1])
        print('url : ')
        print(urls)
        j = j+1
        if j>9 :
            break
        
    return recommend


# 게임 장르와 이름을 결합한 열 생성
def get_important_features(data):
    important_features=[]
    for i in range(0,data.shape[0]):
        important_features.append(data['genre'][i]+' '+data['name'][i])

    return important_features


# Second Recommendations
def make_ratings(id, recommend, games, list_star, max_len=10):
    # user_id 설정
    user_id=[id]*max_len
    
    # user_data 생성
    user_data=pd.DataFrame({'userid':user_id,'name':recommend,'stars':list_star})
    #print(user_data)

    user_data=pd.merge(user_data,games,on='name')
    user_data=user_data[['userid','appid','stars']]
    
    #steam_raw = pd.read_csv("data/game_rating.csv", header=None, names=["userid", "appid", "stars"])
    con=sqlite3.connect('db.sqlite3')
    steam_raw=pd.read_sql("SELECT * FROM game_rating",con)

    #print(steam_raw.columns)

    steam_raw=pd.concat([steam_raw,user_data])  # 값 추가
    #print(steam_raw.dtypes)

    # 값 저장하기
    # steam_raw.to_csv("data/game_ratings.csv",index=False,header=None)
    steam_raw=steam_raw[steam_raw['stars']!=0]
    steam_raw.to_sql("game_rating",con,if_exists="replace",index=False)

    con.close()
    return steam_raw


def second_train_data():
    # 저장했던 데이터 불러오기
    reader = Reader(line_format='user item rating', sep=',',rating_scale=(0.5, 5))
    data_folds=DatasetAutoFolds(ratings_file='data/game_ratings.csv',reader=reader)

    #ratings = pd.read_csv('data/game_ratings.csv',sep=',',names=['userID','appid','stars'])

    return data_folds


# 플레이하지 않은 게임 불러오기
def get_unplay_surprise(ratings, games, userID):
    seen_games=ratings[ratings['userid']==userID]['appid'].tolist()

    total_games=games['appid'].tolist()

    unseen_games=[game for game in total_games if game not in seen_games]
    print(f'특정 {userID}번 유저가 플레이 한 게임 수: {len(seen_games)}\n')
    print(f'추천할 게임 개수: {len(unseen_games)}\n')
    print(f'전체 게임 수 : {len(total_games)}')

    return unseen_games


# 알고리즘 적용 후 게임 추천
def recommend_game_by_surprise(algo, userID, unseen_games, games, top_n=10):
    #games=games.astype({'appid':int})
    predictions=[algo.predict(str(userID),str(appid)) for appid in unseen_games]
    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)
    top_predictions=predictions[:top_n]
    #print(top_predictions[0])
    #print(games.dtypes)

    top_game_ids=[int(pred.iid) for pred in top_predictions]
    top_game_ratings=[pred.est.round(2) for pred in top_predictions]
    top_game_titles=games[games.appid.isin(top_game_ids)]['name']
    top_game_desc=games[games.appid.isin(top_game_ids)]['desc_snippet']
    top_game_url=games[games.appid.isin(top_game_ids)]['url']
    #print(top_game_titles)

    top_game_preds=[(ids,rating,title,desc,url) for ids, rating, title, desc, url in zip(top_game_ids,top_game_ratings,top_game_titles,top_game_desc,top_game_url)]
    #print(top_game_preds)

    return top_game_preds

def second_game_recommend(data_folds, ratings, games):
    # 전부 훈련 데이터로 사용함
    trainset=data_folds.build_full_trainset()
    algo=SVD()
    algo.fit(trainset)

    unseen_lst=get_unplay_surprise(ratings,games,1)
    top_game_preds=recommend_game_by_surprise(algo,1,unseen_lst,games,top_n=20)
    print(top_game_preds)
    print()
    print('#'*8,'Top-20 추천게임 리스트','#'*8)
    game_cnt=1
    for top_game in top_game_preds:
        print('* ',game_cnt,'번째 추천 게임')
        print('* 추천 게임 이름: ', top_game[2])
        print('* 해당 게임의 예측평점: ', top_game[1])
        print('* 해당 게임의 설명: ', top_game[3])
        print('* 해당 게임의 url: ', top_game[4])
        print()
        game_cnt=game_cnt+1

    return top_game_preds
