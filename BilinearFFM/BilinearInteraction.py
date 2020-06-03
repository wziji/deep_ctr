import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import itertools
from tensorflow.keras.initializers import (Zeros, glorot_normal, glorot_uniform)
from tensorflow.keras.layers import Concatenate


class BilinearInteraction(Layer):
    """
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Arguments
        - **str** : String, types of bilinear functions used in this layer.

        - **seed** : A Python integer to use as random seed.
    """

    def __init__(self, bilinear_type="each", seed=1024, **kwargs):
        self.bilinear_type = bilinear_type
        self.seed = seed

        super(BilinearInteraction, self).__init__(**kwargs)

    def build(self, input_shape):

        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called on a list of at least 2 inputs')
        embedding_size = int(input_shape[0][-1])

        if self.bilinear_type == "all":
            self.W = self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(len(input_shape) - 1)]
        elif self.bilinear_type == "interaction":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(len(input_shape)), 2)]
        else:
            raise NotImplementedError

        super(BilinearInteraction, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs[0]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        if self.bilinear_type == "all":
            p = [tf.multiply(tf.tensordot(v_i, self.W, axes=(-1, 0)), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]
        elif self.bilinear_type == "each":
            p = [tf.multiply(tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError

        return Concatenate(axis=-1)(p)
        

    def compute_output_shape(self, input_shape):
        filed_size = len(input_shape)
        embedding_size = input_shape[0][-1]

        return (None, 1, filed_size * (filed_size - 1) // 2 * embedding_size)

    def get_config(self, ):
        config = {'bilinear_type': self.bilinear_type, 'seed': self.seed}
        base_config = super(BilinearInteraction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
