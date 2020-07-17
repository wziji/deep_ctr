import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, concatenate, Flatten, Dense, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from transformer import *


def bst_model(sparse_input_length = 1, \
    max_seq_length = 50, \
    vocab_size_dict = None, \
    embedding_dim = 64, \
    dnn_unit_list = [2056, 512, 32], \
    activation = 'relu', \
    dropout = 0.2, \
    n_layers = 2, \
    num_heads = 8, \
    middle_units = 1024
    ):
    
    
    # 1. Input layer
    
    user_id_input_layer = Input(shape=(sparse_input_length, ), name="user_id_input_layer")
    gender_input_layer = Input(shape=(sparse_input_length, ), name="gender_input_layer")
    age_input_layer = Input(shape=(sparse_input_length, ), name="age_input_layer")
    
    
    user_click_item_seq_input_layer = Input(shape=(max_seq_length, ), name="user_click_item_seq_input_layer")
    user_click_cate_seq_input_layer = Input(shape=(max_seq_length, ), name="user_click_cate_seq_input_layer")
    
    
    item_input_layer = Input(shape=(sparse_input_length, ), name="item_input_layer")
    cate_input_layer = Input(shape=(sparse_input_length, ), name="cate_input_layer")
    
    
    padding_mask_input = concatenate([user_click_item_seq_input_layer, \
                                item_input_layer], axis=-1)
    
    
    
    # 2. Embedding layer
    
    user_id_embedding_layer = Embedding(vocab_size_dict["user_id"]+1, embedding_dim, \
                                        mask_zero=True, name='user_id_embedding_layer')(user_id_input_layer)
    gender_embedding_layer = Embedding(vocab_size_dict["gender"]+1, embedding_dim, \
                                       mask_zero=True, name='gender_embedding_layer')(gender_input_layer)
    age_embedding_layer = Embedding(vocab_size_dict["age"]+1, embedding_dim, \
                                    mask_zero=True, name='age_embedding_layer')(age_input_layer)
    
    
    item_id_embedding = Embedding(vocab_size_dict["item_id"]+1, embedding_dim, \
                                mask_zero=True, name='item_id_embedding')
    cate_id_embedding = Embedding(vocab_size_dict["cate_id"]+1, embedding_dim, \
                                mask_zero=True, name='cate_id_embedding')
    
    user_click_item_seq_embedding_layer = item_id_embedding(user_click_item_seq_input_layer)
    user_click_cate_seq_embedding_layer = cate_id_embedding(user_click_cate_seq_input_layer)
    
    target_item_embedding_layer = item_id_embedding(item_input_layer)
    target_cate_embedding_layer = cate_id_embedding(cate_input_layer)
    

    
    # 3. Concat layer
    
    other_features_concat_layer = concatenate([user_id_embedding_layer, gender_embedding_layer, \
                                               age_embedding_layer], axis=-1)
    
    other_features_concat_layer = tf.reshape(other_features_concat_layer, \
                                             (-1, other_features_concat_layer.shape[1] * other_features_concat_layer.shape[2]))
    
    user_history_sequence_target_item_concat_layer = concatenate([user_click_item_seq_embedding_layer, \
                                                                  target_item_embedding_layer], axis=1)
    
    user_history_sequence_target_cate_concat_layer = concatenate([user_click_cate_seq_embedding_layer, \
                                                                  target_cate_embedding_layer], axis=1)
    
    input_transformer_layer = concatenate([user_history_sequence_target_item_concat_layer, \
                                           user_history_sequence_target_cate_concat_layer], axis=-1)
    
    
    # 4. Transformer layer

    seq_len = max_seq_length + 1
    d_model = input_transformer_layer.shape[-1]
    
    padding_mask_list = padding_mask(padding_mask_input)
    
    output_tranformer_layer = Encoder(n_layers, d_model, num_heads, 
                                middle_units, seq_len)(input_transformer_layer, padding_mask_list, False)
    
    output_tranformer_layer = tf.reshape(output_tranformer_layer, \
                                         (-1, output_tranformer_layer.shape[1] * output_tranformer_layer.shape[2]))

    
    # 5. DNN layer
    input_dnn_layer = concatenate([output_tranformer_layer, other_features_concat_layer], \
                                 axis=-1)
    
    for inx in range(len(dnn_unit_list)):
        input_dnn_layer = Dense(dnn_unit_list[inx], activation=activation, \
                                name="FC_{0}".format(inx+1))(input_dnn_layer)
        
        input_dnn_layer = Dropout(dropout, name="dropout_{0}".format(inx+1))(input_dnn_layer)
        
    
    output = Dense(1, activation='sigmoid', \
                    name='Sigmoid_output_layer')(input_dnn_layer)
    
    
    
    # Output model
    inputs_list = [user_id_input_layer, gender_input_layer, age_input_layer, \
                        user_click_item_seq_input_layer, user_click_cate_seq_input_layer, \
                        item_input_layer, cate_input_layer]
    
    model = Model(inputs = inputs_list,
                  outputs = output)
    
    
    return model
    
    
    
if __name__ == "__main__":
    vocab_size_dict = {
    "user_id": 300,
    "gender": 2,
    "age": 10,
    "item_id": 5000,
    "cate_id": 213}

    bst_model = bst_model(vocab_size_dict=vocab_size_dict)

    print(bst_model.summary())
