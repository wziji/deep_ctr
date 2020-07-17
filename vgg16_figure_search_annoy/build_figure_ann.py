# encoding:utf-8
from annoy import AnnoyIndex
import pickle
import numpy as np
np.random.seed(20200601)
import sys, time
from tqdm import tqdm

def build_ann(name_path=None, \
       vec_path=None, \
       index_to_name_dict_path=None, \
       ann_path=None, \
       dim=64, \
       n_trees=10):

    name_path = open(name_path, 'rb')
    vec_path = open(vec_path, 'rb')
    img_name_list = pickle.load(name_path)
    img_vec_list = pickle.load(vec_path)
    

    ann = AnnoyIndex(dim)
    idx = 0
    batch_size = 100 * 10000
    index_to_name_dict = {}
    

    for name, vec in tqdm(zip(img_name_list, img_vec_list)):
        ann.add_item(idx, vec)
        index_to_name_dict[idx] = name

        idx += 1
        if idx % batch_size == 0:
            print("%s00w" % (int(idx/batch_size)))

    print("Add items Done!\nStart building trees")

    ann.build(n_trees)
    print("Build Trees Done!")
    
    ann.save(ann_path)
    print("Save ann to %s Done!" % (ann_path))

    fd = open(index_to_name_dict_path, 'wb')
    pickle.dump(index_to_name_dict, fd)
    fd.close()
    print("Saving index_to_name mapping Done!")


if __name__ == '__main__':
    name_path = "img_name_list.pkl"
    vec_path = "img_feature_list.pkl"
    index_to_name_dict_path = "index_to_name_dict.pkl"
    ann_path = "img_feature_list.ann"
    dim = 25088
    n_trees = 10

    build_ann(name_path=name_path, \
        vec_path=vec_path, \
        index_to_name_dict_path=index_to_name_dict_path, \
        ann_path=ann_path, \
        dim=dim, \
        n_trees=n_trees)