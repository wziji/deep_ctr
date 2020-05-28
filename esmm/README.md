# tf.version == '2.1.0'

# Data format : 

```python
说明：

（1）第1列：ctr 的值，0表示曝光后未点击，1表示曝光后点击；
（2）第2列：ctcvr 的值，0表示（曝光后未点击、或曝光后点击）后未购买，1表示（曝光、点击）后购买；

（3）第3-130列：用户的预训练Embedding；
（4）第131-258列：商品的预训练Embedding；

```


# run model
```shell
sh master.sh

```


# 参考

```python
1. https://mp.weixin.qq.com/s/J3GH1G1xuzEaPdxLzKDGOA
```