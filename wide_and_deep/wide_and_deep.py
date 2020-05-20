import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def wide_and_deep(
    sparse_feature_list=[],
    sparse_feature_vocabulary_size_list=[],
    sparse_input_length=1,
    
    dense_feature_list=[],
    dense_input_length=1,
    
    embedding_dim = 64,
    hidden_unit_list = [128, 64], 
    classification = 3
    ):
 

    # 1. Input layer
    input_layer_dict = {}
    input_layer_list = []
    
    
    # 1.1 finetune feature
    user_finetune_embedding = Input(shape=(128, ), dtype = 'float32', name="user_finetune_embedding")
    input_layer_list.append(user_finetune_embedding)
    input_layer_dict["user_finetune_embedding"] = tf.expand_dims(user_finetune_embedding, 1)
    
    
    # 1.2 sparse feature
    for f in sparse_feature_list:
        input_layer_dict[f] = Input(shape=(sparse_input_length, ), name=f+"_input_layer")
        input_layer_list.append(input_layer_dict[f])
        
        
    # 1.3 dense feature
    for f in dense_feature_list:
        row_dense_input_layer = Input(shape=(dense_input_length, ), name=f+"_input_layer")
        input_layer_list.append(row_dense_input_layer)
        input_layer_dict[f] = tf.expand_dims(row_dense_input_layer, 1)
    
    
        
    # 2. Embedding
    embedding_layer_dict = {}
    
    # 2.1 sparse feature embedding
    for f, v in zip(sparse_feature_list, sparse_feature_vocabulary_size_list):
        embedding_layer_dict[f] = Embedding(v+1, embedding_dim, mask_zero=True, \
                                            name=f+'_embedding_layer')(input_layer_dict[f])
    
    
    # 2.2 concat
    concat_layer = concatenate([input_layer_dict["user_finetune_embedding"]] + \
                                [embedding_layer_dict[i] for i in sparse_feature_list] + \
                                [input_layer_dict[j] for j in dense_feature_list], \
                               axis=-1)
    


    # 3. Linear part
    linear_part = Dense(1, activation='linear')(concat_layer)
    print("linear_part: ", linear_part)
    
    
    
    # 4. Deep part
    deep_part = concat_layer
    
    for i, u in enumerate(hidden_unit_list):
        deep_part = Dense(u, activation="relu", name="FC_{0}".format(i+1))(deep_part)
        deep_part = Dropout(0.3)(deep_part)
        
    print("deep_part: ", deep_part)
    print("\n" * 3)

    
    
    # Output
    output = tf.keras.layers.concatenate([linear_part, deep_part], axis=-1)
    output = Dense(classification, activation="softmax")(output)
    
    
    model = Model(inputs = input_layer_list, \
                  outputs = output)


    return model



if __name__ == "__main__":

    sparse_feature_list = ["user_id", "gender", "age", "item_id"]
    sparse_feature_vocabulary_size_list = [100, 2, 10, 500]
    dense_feature_list = ["click_count", "sales_count"]
    classification = 10

    model = wide_and_deep(sparse_feature_list=sparse_feature_list, \
                     sparse_feature_vocabulary_size_list = sparse_feature_vocabulary_size_list, \
                     dense_feature_list = dense_feature_list, \
                     classification = classification)
    
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='wide_and_deep_model.png', show_shapes=True)
    

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks = [early_stopping_cb]

    
    """
    model.fit(X_train_input, y_train, batch_size=50, epochs=100, 
          callbacks = callbacks,
          validation_data = (X_val_input, y_val))
          
    """