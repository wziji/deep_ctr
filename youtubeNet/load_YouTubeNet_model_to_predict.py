#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model

from YouTubeNet import YouTubeNet
from data_generator import init_output


# 1. Load model

re_model = YouTubeNet()
re_model.load_weights('YouTubeNet_model.h5')




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



"""
(6040, 1, 64)
(3707, 1, 64)

[[0.         0.84161407 0.5913373  1.4273984  0.3627409  0.3708319
  0.         0.         1.1993251  2.023305   0.         0.
  0.         0.         1.7670951  0.558543   1.0881244  1.7819335
  0.6492757  2.6123888  0.3125449  0.36506268 0.         1.1256831
  4.410721   1.7535956  0.52042466 1.4845431  0.4248005  0.
  2.1689777  1.296214   1.1852415  0.         0.         0.43460703
  1.927466   5.7313547  0.         0.         0.         0.36566824
  2.012046   0.         0.         1.5223947  3.8016186  0.
  0.34814402 1.909086   1.8206354  0.39664558 1.0465539  0.
  1.8064818  0.         1.3177121  0.5385138  0.         2.6539533
  0.         0.         0.         0.        ]
 [0.8107976  1.1632944  0.         0.53690577 1.0428483  1.2018232
  3.4726145  2.21235    0.         0.1572555  0.97843236 0.
  0.         0.99380946 0.76257807 0.05231025 1.6611706  0.0405544
  0.9629851  1.3969578  1.9982753  0.         0.1676663  0.
  0.         0.07090688 2.1441605  0.5842841  0.09379    0.
  0.         0.         0.49283475 2.134187   0.         0.8167961
  0.         0.         1.8054122  0.         0.         1.266642
  2.730833   0.         0.         0.5958151  0.         1.2587492
  0.08325796 0.         0.22326717 0.6559374  0.54102665 0.
  1.0489423  0.         0.5308376  0.62447524 0.         0.
  2.3295872  0.         2.5632188  1.3600256 ]]
(3707, 64)


"""