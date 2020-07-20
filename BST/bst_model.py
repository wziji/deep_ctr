import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, concatenate, Flatten, Dense, Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from transformer import Encoder, padding_mask
from din import DinAttentionLayer, din_padding_mask



def bst_model(sparse_input_length = 1, \
    max_seq_length = 50, \
    vocab_size_dict = None, \
    embedding_dim = 512, \
    dnn_unit_list = [512, 128, 32], \
    activation = 'relu', \
    dropout_rate = 0.2, \
    n_layers = 2, \
    num_heads = 8, \
    middle_units = 1024, \
    training = False
    ):
    
    
    # 1. Input layer
    
    # 1.1 user 
    user_id_input_layer = Input(shape=(sparse_input_length, ), name="user_id_input_layer")
    gender_input_layer = Input(shape=(sparse_input_length, ), name="gender_input_layer")
    age_input_layer = Input(shape=(sparse_input_length, ), name="age_input_layer")
    
    
    user_click_item_seq_input_layer = Input(shape=(max_seq_length, ), name="user_click_item_seq_input_layer")
    user_click_cate_seq_input_layer = Input(shape=(max_seq_length, ), name="user_click_cate_seq_input_layer")
    
    
    # 1.2 item
    item_input_layer = Input(shape=(sparse_input_length, ), name="item_input_layer")
    cate_input_layer = Input(shape=(sparse_input_length, ), name="cate_input_layer")
    
    
    
    # 2. Embedding layer
    
    # 2.1 user
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
    
    
    # 2.2 item 
    target_item_embedding_layer = item_id_embedding(item_input_layer)
    target_cate_embedding_layer = cate_id_embedding(cate_input_layer)
    

    
    # 3. Concat layer
    
    # 3.1 user: other features
    other_features_concat_layer = concatenate([user_id_embedding_layer, gender_embedding_layer, \
                                               age_embedding_layer], axis=-1)
    
    
    # 3.1 user: sequence features
    input_transformer_layer = concatenate([user_click_item_seq_embedding_layer, \
                                           user_click_cate_seq_embedding_layer], axis=-1)
    
    
    # 3.2 item
    input_din_layer_query = concatenate([target_item_embedding_layer, \
                                         target_cate_embedding_layer], axis=-1)

    
    # 4. Transformer layer

    d_model = input_transformer_layer.shape[-1]
    padding_mask_list = padding_mask(user_click_item_seq_input_layer)
    #print("padding_mask_list.shape: ", padding_mask_list.shape)
    
    output_tranformer_layer = Encoder(n_layers, d_model, num_heads, 
                                middle_units, max_seq_length, training)([input_transformer_layer, padding_mask_list])

    #print("output_tranformer_layer.shape: ", output_tranformer_layer.shape)

    
    
    # 5. Din attention layer
    
    query = input_din_layer_query
    keys = output_tranformer_layer
    vecs = output_tranformer_layer
    
    din_padding_mask_list = din_padding_mask(user_click_item_seq_input_layer)
    #print("din_padding_mask_list.shape: ", din_padding_mask_list.shape)

    output_din_layer = DinAttentionLayer(d_model, middle_units, dropout_rate)([query, keys, vecs, din_padding_mask_list])
    #print("output_din_layer.shape: ", output_din_layer.shape)
    
    
    
    # 6. DNN layer
    input_dnn_layer = concatenate([other_features_concat_layer, output_din_layer], \
                                 axis=-1)
    
    input_dnn_layer = tf.squeeze(input=input_dnn_layer, axis=[1])
    
    
    for inx in range(len(dnn_unit_list)):
        input_dnn_layer = Dense(dnn_unit_list[inx], activation=activation, \
                                name="FC_{0}".format(inx+1))(input_dnn_layer)
        
        input_dnn_layer = Dropout(dropout_rate, name="dropout_{0}".format(inx+1))(input_dnn_layer)
        
    
    output = Dense(1, activation='sigmoid', \
                   name='Sigmoid_output_layer')(input_dnn_layer)
    
    
    
    # Output model
    
    inputs_list = [user_id_input_layer, gender_input_layer, age_input_layer, \
                   user_click_item_seq_input_layer, user_click_cate_seq_input_layer, \
                   item_input_layer, cate_input_layer]
    
    model = Model(inputs = inputs_list, outputs = output)
    
    
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
    
    plot_model(bst_model, to_file='bst_model.png')

