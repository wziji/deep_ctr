import sys
import datetime
import tarfile
from hashlib import md5
import time
import os


def tar(input_paths, output_path):
	t = tarfile.open(output_path + ".tar.gz", "w:gz")
	

	for input_path in input_paths:
		t.add(input_path, arcname = input_path.split('/')[-1])
	t.close()


def md5_file(name):
	m = md5()
	a_file = open(name, 'rb')
	m.update(a_file.read())
	a_file.close()
	return m.hexdigest()


if __name__ == '__main__':
	model_dir = sys.argv[1]

	input_paths = []

	for model_path in os.listdir(model_dir):
		if model_path.endswith('.tar.gz') or model_path.endswith('.md5'):
			continue
		input_paths.append(model_dir + '/' + model_path)
	

	now_time = datetime.datetime.now().strftime('%Y%m%d%H%m')
	
	tar_model_path = './__model/' + now_time
	md5_path = tar_model_path + '.md5'

	# tar
	print("Staring to tar %s" % input_paths)
	start = time.time()	
	tar(input_paths, tar_model_path)
	print("Tar to %s done! Lasts %.2fs" % (tar_model_path, time.time() - start))

	# generate md5
	start = time.time()
	md5_str = md5_file(tar_model_path + '.tar.gz')
	with open(md5_path, 'w', encoding='utf-8') as f:
		f.write(md5_str)
	print("Write md5 to %s done! Lasts %.2fs" % (md5_path, time.time() - start))

