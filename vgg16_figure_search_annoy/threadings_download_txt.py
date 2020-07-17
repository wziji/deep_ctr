# http://www.uml.org.cn/python/201901221.asp


import threading
import urllib.request
import time


def download_image(url, filename):
	print("download txt from {}".format(url))

	urllib.request.urlretrieve(url, filename)
	print("download done!")


def execute_thread(i):
	textname = "temp/jiari-{}.txt".format(i)

	download_image("http://tool.bitefu.net/jiari/data/2017.txt", textname)


def main():
	threads = []

	for i in range(10):
		thread = threading.Thread(target=execute_thread, args=(i,))
		threads.append(thread)
		thread.start()

	for i in threads:
		i.join()


main()