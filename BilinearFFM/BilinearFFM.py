#-*- coding:utf-8 -*-


import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, concatenate, Dense, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from BilinearInteraction import BilinearInteraction


def BilinearFFM(
    sparse_input_length=1,
    embedding_dim = 64
    ):

    # 1. Input layer
    user_id_input_layer = Input(shape=(sparse_input_length, ), name="user_id_input_layer")
    gender_input_layer = Input(shape=(sparse_input_length, ), name="gender_input_layer")
    age_input_layer = Input(shape=(sparse_input_length, ), name="age_input_layer")
    occupation_input_layer = Input(shape=(sparse_input_length, ), name="occupation_input_layer")
    zip_input_layer = Input(shape=(sparse_input_length, ), name="zip_input_layer")
    item_input_layer = Input(shape=(sparse_input_length, ), name="item_input_layer")

    
    # 2. Embedding layer
    user_id_embedding_layer = Embedding(6040+1, embedding_dim, mask_zero=True, name='user_id_embedding_layer')(user_id_input_layer)
    gender_embedding_layer = Embedding(2+1, embedding_dim, mask_zero=True, name='gender_embedding_layer')(gender_input_layer)
    age_embedding_layer = Embedding(7+1, embedding_dim, mask_zero=True, name='age_embedding_layer')(age_input_layer)
    occupation_embedding_layer = Embedding(21+1, embedding_dim, mask_zero=True, name='occupation_embedding_layer')(occupation_input_layer)
    zip_embedding_layer = Embedding(3439+1, embedding_dim, mask_zero=True, name='zip_embedding_layer')(zip_input_layer)
    item_id_embedding_layer = Embedding(3706+1, embedding_dim, mask_zero=True, name='item_id_embedding_layer')(item_input_layer)
  

    sparse_embedding_list = [user_id_embedding_layer, gender_embedding_layer, age_embedding_layer, \
                             occupation_embedding_layer, zip_embedding_layer, item_id_embedding_layer]
    


    # 3. Bilinear FFM
    bilinear_out = BilinearInteraction()(sparse_embedding_list)


    # Output
    dot_output = tf.nn.sigmoid(tf.reduce_sum(bilinear_out, axis=-1))

    sparse_input_list = [user_id_input_layer, gender_input_layer, age_input_layer, occupation_input_layer, zip_input_layer, item_input_layer]
    model = Model(inputs = sparse_input_list,
                  outputs = dot_output)
    
    

    return model