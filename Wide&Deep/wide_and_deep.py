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
    

    # 1.1 sparse feature
    for f in sparse_feature_list:
        input_layer_dict[f] = Input(shape=(sparse_input_length, ), name=f+"_input_layer")
        input_layer_list.append(input_layer_dict[f])
        
        
    # 1.2 dense feature
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
    concat_layer = concatenate([embedding_layer_dict[i] for i in sparse_feature_list] + \
                                [input_layer_dict[j] for j in dense_feature_list], \
                               axis=-1)
    


    # 3. Linear part
    linear_part = Dense(1, activation='linear')(concat_layer)

    
    
    # 4. Deep part
    deep_part = concat_layer
    
    for i, u in enumerate(hidden_unit_list):
        deep_part = Dense(u, activation="relu", name="FC_{0}".format(i+1))(deep_part)
        deep_part = Dropout(0.3)(deep_part)

    
    
    # Output
    output = tf.keras.layers.concatenate([linear_part, deep_part], axis=-1)
    
    # Multi-classification
    if classification > 2: 
        output = Dense(classification, activation="softmax")(output)
        
    # Binary-classification
    else:
        output = Dense(1, activation="sigmoid")(output)
    
    
    model = Model(inputs = input_layer_list, \
                  outputs = output)


    return model



if __name__ == "__main__":

    sparse_feature_list = ["user_id", "gender", "age", "item_id"]
    sparse_feature_vocabulary_size_list = [100, 2, 10, 500]
    dense_feature_list = ["click_count", "sales_count"]
    classification = 2

    model = wide_and_deep(sparse_feature_list=sparse_feature_list, \
                     sparse_feature_vocabulary_size_list = sparse_feature_vocabulary_size_list, \
                     dense_feature_list = dense_feature_list, \
                     classification = classification)
    
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='wide_and_deep_model.png', show_shapes=True)
    

    # Multi-classification
    if classification > 2:
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    # Binary-classification
    else:
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks = [early_stopping_cb]

    
    """
    model.fit(X_train_input, y_train, batch_size=50, epochs=100, 
          callbacks = callbacks,
          validation_data = (X_val_input, y_val))
          
    """