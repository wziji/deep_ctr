import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K



def din_padding_mask(seq):
    # 获取为 0的padding项
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 扩充维度用于attention矩阵
    return seq[:, np.newaxis, :] # (batch_size, 1, seq_len)



class LocalActivationUnit(tf.keras.layers.Layer):

    def __init__(self, d_model, middle_units, dropout_rate, **kwargs):
        self.d_model = d_model
        self.middle_units = middle_units
        self.dropout_rate = dropout_rate

        super(LocalActivationUnit, self).__init__(**kwargs)


    def build(self, input_shape):

        self.dnn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.middle_units, activation='relu'),
            tf.keras.layers.Dense(self.d_model, activation='relu')
            ])

        super(LocalActivationUnit, self).build(input_shape)



    def call(self, inputs, training=None, **kwargs):

        query, keys = inputs
        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        att_out = self.dnn(att_input)
        attention_score = tf.keras.layers.Dense(1)(att_out)

        return attention_score


    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)


    def get_config(self):
        config = {'d_model': self.d_model, 'middle_units': self.middle_units, 'dropout_rate': self.dropout_rate}
        base_config = super(LocalActivationUnit, self).get_config()
        
        return dict(list(base_config.items()) + list(config.items()))



# 构造 Din Attention Layer 层

class DinAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, middle_units, dropout_rate, **kwargs):
        super(DinAttentionLayer, self).__init__(**kwargs)

        self.d_model = d_model
        self.middle_units = middle_units
        self.dropout_rate = dropout_rate
        
        self.local_activation_unit = LocalActivationUnit(d_model, middle_units, dropout_rate)
    
    
    def call(self, inputs, **kwargs):
        query, keys, values, mask = inputs
      
        scaled_attention_logits = self.local_activation_unit([query, keys])
        scaled_attention_logits = tf.transpose(scaled_attention_logits, perm=[0, 2, 1])
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)        
        output = tf.matmul(attention_weights, values)
        
        return output
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], 1, input_shape[1][2])


    def get_config(self):
        config = {'d_model': self.d_model, 'use_bias': self.middle_units, 'dropout_rate': self.dropout_rate}
        base_config = super(DinAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



if __name__ == "__main__":
    query = tf.random.uniform((10, 1, 64))
    keys = tf.random.uniform((10, 50, 64))
    vecs = keys
    
    din_padding_mask_list = din_padding_mask(np.random.randint(0, 15, size=(10, 50)))
    print("din_padding_mask_list.shape: ", din_padding_mask_list.shape)

    output = DinAttentionLayer(32, 64, 0.1)([query, keys, vecs, din_padding_mask_list])
    print("output.shape: ", output.shape)
