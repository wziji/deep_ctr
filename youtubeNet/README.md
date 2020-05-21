# tf.__version__ == '2.1.0'


# 运行
``` shell
sh master.sh

```


# YouTubeNet's summary

```python

Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
user_click_item_seq_input_layer [(None, 50)]         0                                            
__________________________________________________________________________________________________
user_id_input_layer (InputLayer [(None, 1)]          0                                            
__________________________________________________________________________________________________
gender_input_layer (InputLayer) [(None, 1)]          0                                            
__________________________________________________________________________________________________
age_input_layer (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
occupation_input_layer (InputLa [(None, 1)]          0                                            
__________________________________________________________________________________________________
zip_input_layer (InputLayer)    [(None, 1)]          0                                            
__________________________________________________________________________________________________
item_id_embedding_layer (Embedd multiple             237248      pos_item_sample_input_layer[0][0]
                                                                 neg_item_sample_input_layer[0][0]
                                                                 user_click_item_seq_input_layer[0
__________________________________________________________________________________________________
user_click_item_seq_length_inpu [(None, 1)]          0                                            
__________________________________________________________________________________________________
user_id_embedding_layer (Embedd (None, 1, 64)        386624      user_id_input_layer[0][0]        
__________________________________________________________________________________________________
gender_embedding_layer (Embeddi (None, 1, 64)        192         gender_input_layer[0][0]         
__________________________________________________________________________________________________
age_embedding_layer (Embedding) (None, 1, 64)        512         age_input_layer[0][0]            
__________________________________________________________________________________________________
occupation_embedding_layer (Emb (None, 1, 64)        1408        occupation_input_layer[0][0]     
__________________________________________________________________________________________________
zip_embedding_layer (Embedding) (None, 1, 64)        220160      zip_input_layer[0][0]            
__________________________________________________________________________________________________
sequence_pooling_layer_2 (Seque (None, 1, 64)        0           item_id_embedding_layer[2][0]    
                                                                 user_click_item_seq_length_input_
__________________________________________________________________________________________________
pos_item_sample_input_layer (In [(None, 1)]          0                                            
__________________________________________________________________________________________________
neg_item_sample_input_layer (In [(None, 10)]         0                                            
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 1, 384)       0           user_id_embedding_layer[0][0]    
                                                                 gender_embedding_layer[0][0]     
                                                                 age_embedding_layer[0][0]        
                                                                 occupation_embedding_layer[0][0] 
                                                                 zip_embedding_layer[0][0]        
                                                                 sequence_pooling_layer_2[0][0]   
__________________________________________________________________________________________________
FC_1 (Dense)                    (None, 1, 128)       49280       concatenate_4[0][0]              
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 11, 64)       0           item_id_embedding_layer[0][0]    
                                                                 item_id_embedding_layer[1][0]    
__________________________________________________________________________________________________
FC_2 (Dense)                    (None, 1, 64)        8256        FC_1[0][0]                       
__________________________________________________________________________________________________
tf_op_layer_transpose_2 (Tensor [(None, 64, 11)]     0           concatenate_5[0][0]              
__________________________________________________________________________________________________
tf_op_layer_MatMul_2 (TensorFlo [(None, 1, 11)]      0           FC_2[0][0]                       
                                                                 tf_op_layer_transpose_2[0][0]    
__________________________________________________________________________________________________
tf_op_layer_Softmax_2 (TensorFl [(None, 1, 11)]      0           tf_op_layer_MatMul_2[0][0]       
==================================================================================================
Total params: 903,680
Trainable params: 903,680
Non-trainable params: 0
__________________________________________________________________________________________________
None

```


# 运行过程

```python

WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
WARNING:tensorflow:sample_weight modes were coerced from
  ...
    to  
  ['...']
Train for 989 steps, validate for 7 steps
Epoch 1/2
989/989 [==============================] - 62s 63ms/step - loss: 1.7179 - sparse_categorical_accuracy: 0.3706 - val_loss: 1.7052 - val_sparse_categorical_accuracy: 0.3857
Epoch 2/2
989/989 [==============================] - 58s 58ms/step - loss: 1.5146 - sparse_categorical_accuracy: 0.4323 - val_loss: 1.5995 - val_sparse_categorical_accuracy: 0.4144
```