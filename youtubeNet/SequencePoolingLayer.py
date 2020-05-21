import tensorflow as tf
from tensorflow.keras.layers import Lambda, Layer


class SequencePoolingLayer(Layer):

    def __init__(self, mode="mean", support_mask=True, sequence_mask_length=50, **kwargs):

        if mode not in ["mean", "max", "sum"]:
            raise ValueError("mode must be `mean`, `max` or `sum` !")

        self.mode = mode
        self.eps = tf.constant(1e-8, tf.float32)
        self.support_mask = support_mask
        self.sequence_mask_length = sequence_mask_length

        super(SequencePoolingLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(SequencePoolingLayer, self).build(input_shape)


    def call(self, input_hist_seq_list, **kwargs):

        hist_user_embedding_list, hist_user_behavior_length = input_hist_seq_list

        if not self.support_mask:

            if self.mode == "max":
                return tf.reduce_max(hist_user_embedding_list, 1, keepdims=True)

            mode_sum = tf.reduce_sum(hist_user_embedding_list, 1, keepdims=True)

            if self.mode == "sum":
                return mode_sum

            if self.mode == "mean":
                return tf.divide(mode_sum, self.sequence_mask_length + self.eps)


        if self.support_mask:

            # mask matrix
            mask_list = tf.sequence_mask(hist_user_behavior_length, self.sequence_mask_length, dtype=tf.float32)

            # transpose mask matrix
            mask_transpose_list = tf.transpose(mask_list, (0, 2, 1))
            embedding_dim = hist_user_embedding_list.shape[-1]

            # expand mask matrix
            mask_tile_list = tf.tile(mask_transpose_list, [1, 1, embedding_dim])


            # max
            if self.mode == "max":
                hist = hist_user_embedding_list - (1-mask_tile_list) * 1e9
                return tf.reduce_max(hist, 1, keepdims=True)


            mode_sum = tf.reduce_sum(hist_user_embedding_list * mask_tile_list, 1, keepdims=True)

            # sum
            if self.mode == "sum":
                return mode_sum

            # mean
            if self.mode == "mean":
                hist_user_behavior_length = tf.reduce_sum(mask_list, axis=-1, keepdims=True)

                return tf.divide(mode_sum, \
                    tf.cast(hist_user_behavior_length, tf.float32) + self.eps)


    def config(self):
        config = {"mode": self.mode, "support_mask": self.support_mask, \
            "sequence_mask_length": self.sequence_mask_length}

        base_config = super(SequencePoolingLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items))