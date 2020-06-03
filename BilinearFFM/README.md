# tf.version == '2.1.0'

# Data format : 

```python
说明：

（1）第1列：user id；
（2）第2列：user gender id；
（3）第3列：user age id；
（4）第4列：user occupation id；
（5）第5列：user zip id；
（6）第6列：item id；
（7）第7列：label；

```


# run model
```shell
sh master.sh

```


# 模型 summary
```python
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
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
item_input_layer (InputLayer)   [(None, 1)]          0
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
item_id_embedding_layer (Embedd (None, 1, 64)        237248      item_input_layer[0][0]
__________________________________________________________________________________________________
bilinear_interaction (BilinearI (None, 1, 960)       20480       user_id_embedding_layer[0][0]
                                                                 gender_embedding_layer[0][0]
                                                                 age_embedding_layer[0][0]
                                                                 occupation_embedding_layer[0][0]
                                                                 zip_embedding_layer[0][0]
                                                                 item_id_embedding_layer[0][0]
__________________________________________________________________________________________________
tf_op_layer_Sum (TensorFlowOpLa [(None, 1)]          0           bilinear_interaction[0][0]
__________________________________________________________________________________________________
tf_op_layer_Sigmoid (TensorFlow [(None, 1)]          0           tf_op_layer_Sum[0][0]
==================================================================================================
Total params: 866,624
Trainable params: 866,624
Non-trainable params: 0
__________________________________________________________________________________________________
None

```


# 参考

```python
1. [FFM及DeepFFM模型在推荐系统的探索](https://zhuanlan.zhihu.com/p/67795161)
2. https://github.com/shenweichen/DeepCTR/blob/master/deepctr/layers/interaction.py
```