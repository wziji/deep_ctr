#-*- coding:utf-8 -*-

# https://github.com/shenweichen/DeepMatch/blob/master/examples/colab_MovieLen1M_YoutubeDNN.ipynb


#! wget http://files.grouplens.org/datasets/movielens/ml-1m.zip -O ./ml-1m.zip 
#! wget https://raw.githubusercontent.com/shenweichen/DeepMatch/master/examples/preprocess.py -O preprocess.py
#! unzip -o ml-1m.zip 


import pandas as pd
import numpy as np

from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model



# 1. Load data
unames = ['user_id','gender','age','occupation','zip']
user = pd.read_csv('ml-1m/users.dat',sep='::',header=None,names=unames)

rnames = ['user_id','movie_id','rating','timestamp']
ratings = pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,names=rnames)

mnames = ['movie_id','title','genres']
movies = pd.read_csv('ml-1m/movies.dat',sep='::',header=None,names=mnames)

data = pd.merge(pd.merge(ratings, movies), user)



print(data.shape)
# (1000209, 10)



# 2. Label Encoding for sparse features, 
# and process sequence features with `gen_date_set` and `gen_model_input`

sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
SEQ_LEN = 50
negsample = 0


features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
feature_max_idx = {}

for feature in features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
item_profile = data[["movie_id"]].drop_duplicates('movie_id')

user_profile.set_index("user_id", inplace=True)
user_item_list = data.groupby("user_id")['movie_id'].apply(list)

train_set, test_set = gen_data_set(data, negsample)
train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)




# 3. Create neg samples

import random
from tqdm import tqdm

train_neg_sample_list = []
test_neg_sample_list = []
all_movie_list = set(data['movie_id'])
neg_sample_num = 10

for i in tqdm(range(len(train_label))):
    a = set(train_model_input['hist_movie_id'][i] + train_model_input['movie_id'][i])
    neg_list = random.sample(list(all_movie_list - a), neg_sample_num)
    train_neg_sample_list.append(np.array(neg_list))
    
for i in tqdm(range(len(test_label))):
    a = set(test_model_input['hist_movie_id'][i] + test_model_input['movie_id'][i])
    neg_list = random.sample(list(all_movie_list - a), neg_sample_num)
    test_neg_sample_list.append(np.array(neg_list))




# 4. Write to .txt

train = open("train.txt", "w")

for i in range(len(train_label)):
    a = train_model_input["user_id"][i]
    b = train_model_input["gender"][i]
    c = train_model_input["age"][i]
    d = train_model_input["occupation"][i]
    e = train_model_input["zip"][i]
    f = train_model_input["hist_movie_id"][i]
    g = train_model_input["hist_len"][i]
    
    h = train_model_input["movie_id"][i]
    m = train_neg_sample_list[i]
    
    train.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"\
               %(str(a), str(b), str(c), str(d), str(e), ','.join([str(ii) for ii in f]), str(g), str(h), ','.join([str(ii) for ii in m])))
    
train.close()



test = open("test.txt", "w")

for i in range(len(test_label)):
    a = test_model_input["user_id"][i]
    b = test_model_input["gender"][i]
    c = test_model_input["age"][i]
    d = test_model_input["occupation"][i]
    e = test_model_input["zip"][i]
    f = test_model_input["hist_movie_id"][i]
    g = test_model_input["hist_len"][i]
    
    h = test_model_input["movie_id"][i]
    m = test_neg_sample_list[i]
    
    test.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"\
               %(str(a), str(b), str(c), str(d), str(e), ','.join([str(ii) for ii in f]), str(g), str(h), ','.join([str(ii) for ii in m])))
    
test.close()
