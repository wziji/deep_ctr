import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

from mind import mind
from data_generator import init_output




# 1. Load model

re_model = mind()
re_model.load_weights('mind_model.h5')

print(re_model.summary())



# 2. Load data

user_id, gender, age, occupation, zip, \
        hist_movie_id, hist_len, pos_movie_id, neg_movie_id = init_output()

with open("test.txt", 'r') as f:
    for line in f.readlines():

        buf = line.strip().split('\t')

        user_id.append(int(buf[0]))
        gender.append(int(buf[1]))
        age.append(int(buf[2]))
        occupation.append(int(buf[3]))
        zip.append(int(buf[4]))
        hist_movie_id.append(np.array([int(i) for i in buf[5].strip().split(",")]))
        hist_len.append(int(buf[6]))
        pos_movie_id.append(int(buf[7]))
        

user_id = np.array(user_id, dtype='int32')
gender = np.array(gender, dtype='int32')
age = np.array(age, dtype='int32')
occupation = np.array(occupation, dtype='int32')
zip = np.array(zip, dtype='int32')
hist_movie_id = np.array(hist_movie_id, dtype='int32')
hist_len = np.array(hist_len, dtype='int32')
pos_movie_id = np.array(pos_movie_id, dtype='int32')




# 3. Generate user features for testing and full item features for retrieval

test_user_model_input = [user_id, gender, age, occupation, zip, hist_movie_id, hist_len]
all_item_model_input = list(range(0, 3706+1))

user_embedding_model = Model(inputs=re_model.user_input, outputs=re_model.user_embedding)
item_embedding_model = Model(inputs=re_model.item_input, outputs=re_model.item_embedding)

user_embs = user_embedding_model.predict(test_user_model_input)
item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

print(user_embs.shape)
print(item_embs.shape)


user_embs = np.reshape(user_embs, (-1, 64))
item_embs = np.reshape(item_embs, (-1, 64))

print(user_embs[:2])
print(item_embs.shape)