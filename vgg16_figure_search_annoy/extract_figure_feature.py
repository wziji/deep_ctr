from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tqdm import tqdm
import pickle
import os

# 1. 加载vgg16模型
model = VGG16(weights='imagenet', include_top=False)
#print(model.summary())


# 2. 提取图片特征
img_path = "figures/"

img_name_list = []
img_feature_list = []

for file in tqdm(os.listdir(img_path)):
    img_name_list.append(file)
    file_path = img_path + file
    
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    img_feature_list.append(features.reshape((7*7*512,)))


# 3. 将图片名称和图片特征保存为pkl格式
f = open("img_feature_list.pkl", 'wb')
pickle.dump(img_feature_list, f)
f.close()

g = open("img_name_list.pkl", 'wb')
pickle.dump(img_name_list, g)
g.close()