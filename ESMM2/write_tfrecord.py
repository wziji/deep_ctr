import tensorflow as tf
import numpy as np


def _parse_line(line):
    buf = line.split(',')
    ctr_label = int(buf[0])
    cvr_label = int(buf[1])
    pin_vec = np.array(buf[2: 128+2], dtype=np.float32)
    sku_vec = np.array(buf[128+2: 128+128+2], dtype=np.float32)

    return pin_vec, sku_vec, ctr_label, cvr_label


def _to_float_feature(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def _to_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(pin_vec, sku_vec, ctr_label, cvr_label):
    feature = {
        'pin_vec': _to_float_feature(pin_vec),
        'sku_vec': _to_float_feature(sku_vec),
        'ctr_label': _to_int64_feature(ctr_label),
        'ctcvr_label': _to_int64_feature(cvr_label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(
        feature=feature))

    example = example_proto.SerializeToString()

    return example


def make_tfrecord(input_path, tf_path):
    writer = tf.data.experimental.TFRecordWriter(tf_path, compression_type="ZLIB")

    def generator():
        with open(input_path, 'r') as f:
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % 10000 == 0:
                    print("write %s lines." % cnt)
                    
                pin_vec, sku_vec, ctr_label, cvr_label = _parse_line(line[:-1])

                yield serialize_example(pin_vec, sku_vec, ctr_label, cvr_label)


    serialized_dataset = tf.data.Dataset.from_generator(
        generator, output_types=tf.string, output_shapes=())

    writer.write(serialized_dataset)

    print("Write tf_path=%s done!" % tf_path)


if __name__ == '__main__':
    dir_path = './'
    train_path = dir_path + 'data/train_data'
    tf_train_path = dir_path + 'data/train_data.tfrecord'
    make_tfrecord(train_path, tf_train_path)

    val_path = dir_path + 'data/val_data'
    tf_val_path = dir_path + 'data/val_data.tfrecord'
    make_tfrecord(val_path, tf_val_path)
    