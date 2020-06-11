# tf.__version__ == '2.1.0'


# 运行
``` shell
sh master.sh

```


# mind's summary

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
sequence_pooling_layer_55 (Sequ (None, 1, 64)        0           item_id_embedding_layer[2][0]    
                                                                 user_click_item_seq_length_input_
__________________________________________________________________________________________________
concatenate_133 (Concatenate)   (None, 1, 384)       0           user_id_embedding_layer[0][0]    
                                                                 gender_embedding_layer[0][0]     
                                                                 age_embedding_layer[0][0]        
                                                                 occupation_embedding_layer[0][0] 
                                                                 zip_embedding_layer[0][0]        
                                                                 sequence_pooling_layer_55[0][0]  
__________________________________________________________________________________________________
tf_op_layer_Tile_2 (TensorFlowO [(None, 5, 384)]     0           concatenate_133[0][0]            
__________________________________________________________________________________________________
capsule_layer_58 (CapsuleLayer) (None, 5, 64)        4346        item_id_embedding_layer[2][0]    
                                                                 user_click_item_seq_length_input_
__________________________________________________________________________________________________
concatenate_134 (Concatenate)   (None, 5, 448)       0           tf_op_layer_Tile_2[0][0]         
                                                                 capsule_layer_58[0][0]           
__________________________________________________________________________________________________
FC_1 (Dense)                    (None, 5, 128)       57472       concatenate_134[0][0]            
__________________________________________________________________________________________________
pos_item_sample_input_layer (In [(None, 1)]          0                                            
__________________________________________________________________________________________________
neg_item_sample_input_layer (In [(None, 10)]         0                                            
__________________________________________________________________________________________________
FC_2 (Dense)                    (None, 5, 64)        8256        FC_1[0][0]                       
__________________________________________________________________________________________________
label_aware_attention_43 (Label (None, 64)           0           FC_2[0][0]                       
                                                                 item_id_embedding_layer[0][0]    
                                                                 user_click_item_seq_length_input_
__________________________________________________________________________________________________
concatenate_135 (Concatenate)   (None, 11, 64)       0           item_id_embedding_layer[0][0]    
                                                                 item_id_embedding_layer[1][0]    
__________________________________________________________________________________________________
tf_op_layer_ExpandDims_7 (Tenso [(None, 1, 64)]      0           label_aware_attention_43[0][0]   
__________________________________________________________________________________________________
tf_op_layer_transpose_34 (Tenso [(None, 64, 11)]     0           concatenate_135[0][0]            
__________________________________________________________________________________________________
tf_op_layer_MatMul_29 (TensorFl [(None, 1, 11)]      0           tf_op_layer_ExpandDims_7[0][0]   
                                                                 tf_op_layer_transpose_34[0][0]   
__________________________________________________________________________________________________
tf_op_layer_Softmax_29 (TensorF [(None, 1, 11)]      0           tf_op_layer_MatMul_29[0][0]      
==================================================================================================
Total params: 916,218
Trainable params: 915,968
Non-trainable params: 250
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
989/989 [==============================] - 137s 139ms/step - loss: 1.6125 - sparse_categorical_accuracy: 0.4041 - val_loss: 1.5422 - val_sparse_categorical_accuracy: 0.4224
Epoch 2/2
989/989 [==============================] - 131s 133ms/step - loss: 1.3553 - sparse_categorical_accuracy: 0.4910 - val_loss: 1.4716 - val_sparse_categorical_accuracy: 0.4604
```


# 参考

[https://github.com/shenweichen/DeepMatch/blob/master/deepmatch/models/mind.py](https://github.com/shenweichen/DeepMatch/blob/master/deepmatch/models/mind.py)

