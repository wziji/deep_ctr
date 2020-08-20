import sys
import numpy as np


def split_train_val(sample_path, train_path, val_path, summary_path, train_percent = None):

	fr = open(sample_path, 'r')
	
	fw_train = open(train_path, 'w')
	fw_val = open(val_path, 'w')

	fw_summary = open(summary_path, 'w')

	n_train = 0
	n_val = 0
	for line in fr:
		r = np.random.random()
		if r < train_percent:
			fw_train.write(line)
			n_train += 1
		else:
			fw_val.write(line)
			n_val +=1 

	summary = 'n_train=' + str(n_train) + ',' + 'n_val=' + str(n_val) 
	fw_summary.write(summary + '\n')


	fr.close()
	fw_train.close()
	fw_val.close()		
	fw_summary.close()

	print('Split train and val done!')
	print('Write summary done!')



if __name__ == '__main__':
	sample_path = sys.argv[1]

	train_path = sys.argv[2]
	val_path = sys.argv[3]
	summary_path = sys.argv[4]
	train_percent = float(sys.argv[5])

	split_train_val(sample_path, train_path, val_path, summary_path, train_percent = train_percent)
	
	
	
