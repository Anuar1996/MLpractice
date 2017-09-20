import numpy as np
import zipfile
import math

def read_data():
    archive = zipfile.ZipFile('ml-1m.zip')
    
    ratings = archive.open('ml-1m/ratings.dat').readlines()
    ratings = preproc_ratings(ratings) 
    
    users = archive.open('ml-1m/users.dat').readlines()
    users = preproc_users(users)
        
    movies = archive.open('ml-1m/movies.dat').readlines()
    movies = preproc_movies(movies)
        
    archive.close()
    return  (users, movies, ratings)

def preproc_ratings(ratings):
    
    res = list()
    
    item_list = list()
    
    i = 0
    for rate in ratings:
        tmp_rate = rate.rstrip().split('::')
        if (i == int(tmp_rate[0]) - 1):
            item_list.append((int(tmp_rate[1]), float(tmp_rate[2]), int(tmp_rate[3])))
        else:
            i = int(tmp_rate[0]) - 1
            res.append(item_list)
            item_list = list()
    return dict(zip(np.arange(6040), res))

def preproc_users(users):
    res = np.empty((len(users), 3)).astype('|S3')
    for user in users:
        tmp_user = user.rstrip().split('::')
        res[int(tmp_user[0]) - 1, 0] = tmp_user[1]
        res[int(tmp_user[0]) - 1, 1] = tmp_user[2]
        res[int(tmp_user[0]) - 1, 2] = tmp_user[3]
    
    n_features = 0
    features_names = []
    
    for i in range(res.shape[1]):
        n_features += np.unique(res[: ,i]).shape[0]
        features_names.append(list(np.unique(res[:, i])))
    res_ = np.zeros((len(users), n_features))
    
    i = 0
    for j in range(3):
        for feature in features_names[j]:
            res_[:, i] = res[:, j] == feature
            i += 1
    del res
    return res_

def preproc_movies(movies):
    movie_types = ["Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
               "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
               "War", "Western"]
    n_features = np.unique(movie_types).shape[0] + 3
    n_movies = 0
    res = np.zeros((len(movies), n_features)).astype('int')
    for i, movie in enumerate(movies):
        tmp_movie = movie.rstrip().split('::')
        movie_name = tmp_movie[1]
        movie_type = tmp_movie[2].split('|')
        movieID = int(tmp_movie[0])
        year = int(movie_name[movie_name.rfind('(') + 1: movie_name.rfind('(') + 5])
        res[i, 0] = movieID
        res[i, 1] = year
        for mov in movie_type:
            j = movie_types.index(mov)
            res[i, j + 3] = 1
        res[i, 2] = res[i, 3:].sum()
    return res

def train_test_split(ratings):
    train_frac = 0.8
    train = []
    test = []
    for u, itemList in ratings.items():
        all = sorted(itemList, key=lambda x: x[2])
        thr = int(math.floor(len(all) * train_frac))
        train.extend(map(lambda x: [u, x[0], x[1] / 5.0], all[:thr]))
        test.extend(map(lambda x: [u, x[0], x[1] / 5.0], all[thr:]))
    train = np.array(train)
    test = np.array(test)
    
    return (train, test)

def construct_train(users, movies, train):
    movie_types = ["Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
               "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
               "War", "Western"]
    n_objects = len(train)
    
    n_features = users.shape[1] + movies.shape[1] + 3

    X = np.zeros((n_objects, n_features))
    
    #constant feature
    left = 0
    right = 1
    X[:, left: right] = 1.0
    
    #users features
    left = right
    right = left + users.shape[1]
    X[:, left: right] = users[train[:, 0].astype('int'), :]
    
    #movies features
    left = right
    right = left + movies.shape[1] - 1
    tmpID = np.zeros(movies[:, 0].max() + 1).astype('int')
    tmpID[movies[:, 0]] = np.arange(movies.shape[0])
    
    X[:, left: right] = movies[tmpID[train[:, 1].astype('int')], 1: ]
    
    #mean rating
    left = right
    right += 1
    mean_rate_users = np.zeros(users.shape[0])
    for userID in range(users.shape[0]):
        indx = np.where(train[:, 0] == userID)[0]
        if indx.shape[0] == 0:
            continue
        mean_rate_users[userID] = train[indx, 2].mean()
    X[:, left: right] = mean_rate_users[train[:, 0].astype('int')][:, np.newaxis]
    
    #movie's rating
    left = right
    right += 1
    mean_rate_movie = np.zeros(movies[:, 0].max() + 1)
    for movieID in range(1, movies[:, 0].max() + 1):
        indx = np.where(train[:, 1] == movieID)[0]
        if indx.shape[0] == 0:
            continue
        mean_rate_movie[movieID] = train[indx, 2].mean()
    X[:, left: right] = mean_rate_movie[train[:, 1].astype('int')][:, np.newaxis]
    
    #f^5
    left = right
    right += 1
    F_5 = np.zeros((len(users), len(movie_types)))
    for userID in range(users.shape[0]):
        u_g = np.zeros(len(movie_types))
        indx_user = np.where(train[:, 0] == userID)[0]
        user_movieID = train[indx_user, 1].astype('int')
        user = train[indx_user, :]
        if user_movieID.shape[0] != 0:
            tmpID2 = np.zeros(user_movieID.max() + 1).astype('int')
        else:
            continue
        tmpID2[user_movieID] = np.arange(user_movieID.shape[0])
          
        a = movies[tmpID[user_movieID]]
        for mov_type in range(len(movie_types)):
            i = a[:, mov_type + 3] == 1
            b = a[i]
            if tmpID2[b[:, 0]].shape[0] == 0:
                continue
            u_g[mov_type] = user[tmpID2[b[:, 0]]][:, 2].mean()
        F_5[userID] = u_g
        X[indx_user, left: right] = ((a[:, 3: ] * u_g).sum(axis=1) / a[:, 2])[:, np.newaxis]

    return (X, mean_rate_users, mean_rate_movie, F_5)

def construct_test(users, movies, test, mean_rate_users, mean_rate_movie, F_5):
    movie_types = ["Action", "Adventure", "Animation", "Children's", "Comedy",
               "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
               "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
               "War", "Western"]
    n_objects = len(test)
    
    n_features = users.shape[1] + movies.shape[1] + 3

    X = np.zeros((n_objects, n_features))
    
    #constant feature
    left = 0
    right = 1
    X[:, left: right] = 1.0
    
    #users features
    left = right
    right = left + users.shape[1]
    X[:, left: right] = users[test[:, 0].astype('int'), :]
    
    #movies features
    left = right
    right = left + movies.shape[1] - 1
    tmpID = np.zeros(movies[:, 0].max() + 1).astype('int')
    tmpID[movies[:, 0]] = np.arange(movies.shape[0])
    
    X[:, left: right] = movies[tmpID[test[:, 1].astype('int')], 1: ]
    
    #mean rating
    left = right
    right += 1
    X[:, left: right] = mean_rate_users[test[:, 0].astype('int')][:, np.newaxis]
    
    #movie's rating
    left = right
    right += 1
    X[:, left: right] = mean_rate_movie[test[:, 1].astype('int')][:, np.newaxis]
    
    #f^5
    left = right
    right += 1
    for userID in range(users.shape[0]):
        indx_user = np.where(test[:, 0] == userID)[0]
        user_movieID = test[indx_user, 1].astype('int')
        user = test[indx_user, :]
        if user_movieID.shape[0] != 0:
            tmpID2 = np.zeros(user_movieID.max() + 1).astype('int')
        else:
            continue
        tmpID2[user_movieID] = np.arange(user_movieID.shape[0])  
        a = movies[tmpID[user_movieID]]
        X[indx_user, left: right] = ((a[:, 3: ] * F_5[userID]).sum(axis=1) / a[:, 2])[:, np.newaxis]

    return X


def MSE(x, y):
    return ((x - y) ** 2 / x.shape[0]).sum()