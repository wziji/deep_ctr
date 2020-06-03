#-*- coding:utf-8 -*-

import numpy as np


def init_output():
    user_id = []
    gender = []
    age = []
    occupation = []
    zip = []
    movie_id = []
    label = []


    return user_id, gender, age, occupation, zip, movie_id, label


def file_generator(input_path, batch_size):

    user_id, gender, age, occupation, zip, movie_id, label = init_output()

    cnt = 0
    
    num_lines = sum([1 for line in open(input_path)])

    while True:

        with open(input_path, 'r') as f:
            for line in f.readlines():

                buf = line.strip().split('\t')

                user_id.append(int(buf[0]))
                gender.append(int(buf[1]))
                age.append(int(buf[2]))
                occupation.append(int(buf[3]))
                zip.append(int(buf[4]))
                movie_id.append(int(buf[5]))
                label.append(int(buf[6]))

                cnt += 1

                if cnt % batch_size == 0 or cnt == num_lines:
                    user_id = np.array(user_id, dtype='int32')
                    gender = np.array(gender, dtype='int32')
                    age = np.array(age, dtype='int32')
                    occupation = np.array(occupation, dtype='int32')
                    zip = np.array(zip, dtype='int32')
                    movie_id = np.array(movie_id, dtype='int32')
                    
                    label = np.array(label, dtype='int32')

                    yield [user_id, gender, age, occupation, zip, movie_id], label

                    user_id, gender, age, occupation, zip, movie_id, label = init_output()

