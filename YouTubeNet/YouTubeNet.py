#-*- coding:utf-8 -*-


import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, concatenate, Dense, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from SequencePoolingLayer import SequencePoolingLayer


def YouTubeNet(
    sparse_input_length=1,
    dense_input_length=1,
    sparse_seq_input_length=50,
    
    embedding_dim = 64,
    neg_sample_num = 10, 
    user_hidden_unit_list = [128, 64]
    ):

    # 1. Input layer
    user_id_input_layer = Input(shape=(sparse_input_length, ), name="user_id_input_layer")
    gender_input_layer = Input(shape=(sparse_input_length, ), name="gender_input_layer")
    age_input_layer = Input(shape=(sparse_input_length, ), name="age_input_layer")
    occupation_input_layer = Input(shape=(sparse_input_length, ), name="occupation_input_layer")
    zip_input_layer = Input(shape=(sparse_input_length, ), name="zip_input_layer")
    
    
    user_click_item_seq_input_layer = Input(shape=(sparse_seq_input_length, ), name="user_click_item_seq_input_layer")
    user_click_item_seq_length_input_layer = Input(shape=(sparse_input_length, ), name="user_click_item_seq_length_input_layer")
    
    
    pos_item_sample_input_layer = Input(shape=(sparse_input_length, ), name="pos_item_sample_input_layer")
    neg_item_sample_input_layer = Input(shape=(neg_sample_num, ), name="neg_item_sample_input_layer")


    
    # 2. Embedding layer
    user_id_embedding_layer = Embedding(6040+1, embedding_dim, mask_zero=True, name='user_id_embedding_layer')(user_id_input_layer)
    gender_embedding_layer = Embedding(2+1, embedding_dim, mask_zero=True, name='gender_embedding_layer')(gender_input_layer)
    age_embedding_layer = Embedding(7+1, embedding_dim, mask_zero=True, name='age_embedding_layer')(age_input_layer)
    occupation_embedding_layer = Embedding(21+1, embedding_dim, mask_zero=True, name='occupation_embedding_layer')(occupation_input_layer)
    zip_embedding_layer = Embedding(3439+1, embedding_dim, mask_zero=True, name='zip_embedding_layer')(zip_input_layer)
    
    item_id_embedding_layer = Embedding(3706+1, embedding_dim, mask_zero=True, name='item_id_embedding_layer')
    pos_item_sample_embedding_layer = item_id_embedding_layer(pos_item_sample_input_layer)
    neg_item_sample_embedding_layer = item_id_embedding_layer(neg_item_sample_input_layer)
    
    user_click_item_seq_embedding_layer = item_id_embedding_layer(user_click_item_seq_input_layer)
    #user_click_item_seq_embedding_layer = SequencePoolingLayer(sequence_mask_length=sparse_seq_input_length)\
    #    ([user_click_item_seq_embedding_layer, user_click_item_seq_length_input_layer])

    user_click_item_seq_embedding_layer  = tf.reduce_mean(user_click_item_seq_embedding_layer, 1, keepdims=True)

    

    ### ********** ###
    # user part
    ### ********** ###

    # 3. Concat "sparse" embedding & "sparse_seq" embedding
    user_embedding_layer = concatenate([user_id_embedding_layer, gender_embedding_layer, age_embedding_layer,
                                       occupation_embedding_layer, zip_embedding_layer, user_click_item_seq_embedding_layer], 
                                       axis=-1)


    for i, u in enumerate(user_hidden_unit_list):
        user_embedding_layer = Dense(u, activation="relu", name="FC_{0}".format(i+1))(user_embedding_layer)
        #user_embedding_layer = Dropout(0.3)(user_embedding_layer)
        
    
    ### ********** ###
    # item part
    ### ********** ###

    item_embedding_layer = concatenate([pos_item_sample_embedding_layer, neg_item_sample_embedding_layer], \
                                       axis=1)
    
    item_embedding_layer = tf.transpose(item_embedding_layer, [0,2,1])
    


    # Output
    dot_output = tf.matmul(user_embedding_layer, item_embedding_layer) 
    dot_output = tf.nn.softmax(dot_output) # 输出11个值，index为0的值是正样本，负样本的索引位置为[1-10]
    
    user_inputs_list = [user_id_input_layer, gender_input_layer, age_input_layer, \
                        occupation_input_layer, zip_input_layer, \
                        user_click_item_seq_input_layer, user_click_item_seq_length_input_layer]
    
    item_inputs_list = [pos_item_sample_input_layer, neg_item_sample_input_layer]

    model = Model(inputs = user_inputs_list + item_inputs_list,
                  outputs = dot_output)
    
    #print(model.summary())
    #tf.keras.utils.plot_model(model, to_file='YouTubeNet_model.png', show_shapes=True)



    model.__setattr__("user_input", user_inputs_list)
    model.__setattr__("user_embedding", user_embedding_layer)
    
    model.__setattr__("item_input", pos_item_sample_input_layer)
    model.__setattr__("item_embedding", pos_item_sample_embedding_layer)
    
    return model