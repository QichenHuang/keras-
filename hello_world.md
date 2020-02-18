# 第一个程序(Hello World)

学过编程的人都知道，Hello World基本是大多数编程之路的开始，通过第一个Hello World的简单程序，
一是检查编程环境是否配置完成；二是给入门的新手程序员一些激励，开始编程的第一步。 

Keras是专门用来构建深度学习的各种网络结构的工具，该第一个程序构建包含一个隐藏层的二分类全连接网络，包含训练，验证，预测等步骤，主要展示代码的大致过程，具体细节部分会在后续篇章依次分享。

## 基本步骤

导入相关包
``` python
from keras.models import Sequential
from keras.layers import Dense
```
构建模型  
Keras通过对各种网络层的连接组合构建完整的神经网络。`Sequential`是一个将网络层依次线性拼接的容器。`Dense`是全连接层。  
这里构建一个`Sequential`模型，输入维度为100，添加一个包含64个单元的全连接层，再添加只有一个单元的输出层
``` python
model = Sequential()
model.add(Dense(units=64,activation='relu',input_dim=100))
model.add(Dense(units=1,activation='sigmoid'))
```
模型编译，设定二元交叉熵损失函数和随机梯度下降优化器，另加一个准确率为评估指标，也可以不添加评估指标
``` python
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
```
模型训练，给定指定尺寸的训练数据和标签，周期为5，批次大小为32
``` python
model.fit(train_data,train_label,epochs=5,batch_size=32)
```
模型评估，若编译时没有设置指标，输出一个loss的标量；若设置了指标，评估函数得出loss和各指标的列表。  
可通过`model.metrics_names`查看评估函数的各个输出的标签
``` python
loss_and_metrics = model.evaluate(train_data, train_label, batch_size=128)
```
模型预测，给定测试数据，模型预测出结果
``` python
classes = model.predict(test_data,batch_size=128)
```
## 输出结果
``` python
>>>loss_and_metrics
[0.6955518779754639, 0.5239999890327454]
>>>model.metrics_names
['loss', 'accuracy']
>>>classes
array([[0.6157647 ],
       [0.38050696],
       ...
       [0.61919147],
       [0.5762338 ]], dtype=float32)
```
需要完整代码的，请查看[源代码](./hello_world.py)  
[返回主页](./README.md)
