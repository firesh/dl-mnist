# 手写数字识别的神经网络

> 《Python 神经网络编程》练习项目

## 数据源
* [训练数据](http://wwww.pjreddie.com/media/files/mnist_train.csv)
* [测试数据](http://wwww.pjreddie.com/media/files/mnist_test.csv)
  
存放在`dataset`目录下。

## 配置参数
```
input_nodes = 784
hidden_nodes = 300
output_nodes = 10
learning_rate = 0.1
epochs = 10
```

## 启动
```
python number.py
```

## 效果
经过10纪元的训练，识别正确率为`97.67%`
```
(py3) MacBook-Pro-Wang:dl-number wang$ python number.py
performance: 0.9767
```
