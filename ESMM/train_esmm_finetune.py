import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Multiply, Dot
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


def parse_example(proto):
	desc = {\
		'pin_vec' : tf.io.FixedLenFeature([128], tf.float32), \
		'sku_vec' : tf.io.FixedLenFeature([128], tf.float32), \
		'ctr_label' : tf.io.FixedLenFeature([], tf.int64), \
		'ctcvr_label' : tf.io.FixedLenFeature([], tf.int64), \
	}

	example = tf.io.parse_single_example(proto, desc)
	
	pin_vec = example['pin_vec']
	sku_vec = example['sku_vec']
	ctr_label = example['ctr_label']
	ctcvr_label = example['ctcvr_label']

	return (pin_vec, sku_vec), (ctr_label, ctcvr_label)
	

def get_tfrecord_dataset(tf_path, batch_size = None, num_parallel_calls = None):
	dataset = tf.data.TFRecordDataset(tf_path, compression_type = "ZLIB")
	print('type(dataset)', type(dataset))
	#dataset = dataset.repeat.shuffle(buffer_size = 1024)\
	dataset = dataset.repeat()\
		.map(parse_example, num_parallel_calls = num_parallel_calls)\
		.batch(batch_size)

	return dataset



def train_finetune(train_path, val_path, model_path, \
	n_train = None, \
	n_val = None):

	model = build_model()
	print(model.summary())


	batch_size = 128
	epochs = 100
	
	train_steps_per_epoch = int(n_train / batch_size)
	val_steps_per_epoch = int(n_val / batch_size)
	
	num_parallel_calls = 4

	train_tfrecord_dataset = get_tfrecord_dataset(\
		train_path, \
		batch_size = batch_size, \
		num_parallel_calls = num_parallel_calls)
	
	val_tfrecord_dataset = get_tfrecord_dataset(\
		val_path, \
		batch_size = batch_size, \
		num_parallel_calls = num_parallel_calls)


	early_stopping_cb = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
	
	tensorboard_cb = TensorBoard(\
		log_dir = './logs', \
		histogram_freq = 0, \
		write_graph = True, \
		write_grads = True, \
		write_images = True)
		
	

	callbacks = [early_stopping_cb, tensorboard_cb]


	start = time.time()

	history = model.fit(\
		train_tfrecord_dataset, \
		steps_per_epoch = train_steps_per_epoch, \
		epochs = epochs, \
		verbose = 1, \
		callbacks = callbacks, \
		validation_data = val_tfrecord_dataset, \
		validation_steps = val_steps_per_epoch, \
		max_queue_size = 10, \
		workers = 1, \
		use_multiprocessing = False, \
		shuffle = True, \
		initial_epoch = 0)

	model.save_weights(model_path)

	last = time.time() - start
	print("Train model to %s done! Lasts %.2fs" % (model_path, last))




def build_model():
	n_pin_vec = 128
	n_sku_vec = 128

	pin_vec = Input(shape=(n_pin_vec, ), dtype = 'float32')
	sku_vec = Input(shape=(n_sku_vec, ), dtype = 'float32')

	# ctr_part
	ctr_pin_part = Dense(64, activation='relu')(pin_vec)
	ctr_sku_part = Dense(64, activation='relu')(sku_vec)
	
	ctr_prod = Multiply()([ctr_pin_part, ctr_sku_part])
	ctr_prob = Dense(1, activation='sigmoid', name='ctr_prob')(ctr_prod)

	
	# ctcvr_part
	cvr_pin_part = Dense(64, activation='relu')(pin_vec)
	cvr_sku_part = Dense(64, activation='relu')(sku_vec)
	
	cvr_prod = Multiply()([cvr_pin_part, cvr_sku_part])
	cvr_prob = Dense(1, activation='sigmoid', name='cvr_prob')(cvr_prod)

	#ctcvr_prob = ctr_prob * cvr_prob
	ctcvr_prob = Multiply(name = 'ctcvr_prob')([ctr_prob, cvr_prob])

	model = Model(inputs = [pin_vec, sku_vec], outputs = [ctr_prob, ctcvr_prob])

    
	model.compile(optimizer = 'adam', \
		loss = {'ctr_prob' : 'binary_crossentropy', 'ctcvr_prob' : 'binary_crossentropy'}, \
		#metrics = {'ctr_prob' : 'accuracy', 'ctcvr_prob' : 'accuracy'})
		metrics = ['accuracy'])

	return model
	


if __name__ == '__main__':

	train_path = sys.argv[1]
	val_path = sys.argv[2]
	model_path = sys.argv[3]

	train_val_summary_path = sys.argv[4]

	n_train = 0
	n_val = 0
	fr = open(train_val_summary_path, 'r')
	for line in fr:
		buf = line[:-1].split(',')
		n_train = int(buf[0].split('=')[1])
		n_val = int(buf[1].split('=')[1])
		break
	fr.close()

	train_finetune(train_path, val_path, model_path, \
		n_train = n_train, \
		n_val = n_val)

	
