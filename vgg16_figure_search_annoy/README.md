# vgg16 model extract figure feature, Annoy to search similar figures

### 推荐阅读：[Annoy最近邻检索技术之 “图片检索”](https://zhuanlan.zhihu.com/p/148819536)


### 1. 创建一个文件夹用于保存图片

> mkdir figures


### 2. 下载数据

> python download_jd_figure.py


### 3. 提取图片特征

[vgg16 官方下载](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5)

[百度网盘中下载](https://pan.baidu.com/s/1Exa8g_q9hVmqOU9SBrIxrg) ，提取码：qtsb

将 vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 模型放入 ~/.keras/models 路径中即可。

> python extract_figure_feature.py


### 4. 基于图片向量特征，构建ANN树
> python build_figure_ann.py


### 5. 搜索topN相似图片
> python search_topN_figure.py
