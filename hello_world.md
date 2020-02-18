# 第一个程序(Hello World)

学过编程的人都知道，Hello World基本是大多数编程之路的开始，通过第一个Hello World的简单程序，
一是检查编程环境是否配置完成；二是给刚入门的新手程序员一些激励，开始编程的第一步。  

Keras是专门用来构建深度学习的各种网络结构的工具，该第一个程序构建包含一个隐藏层的二分类全连接网络，包含训练，验证，预测等步骤。

## 基本步骤

导入相关包
``` python
from keras.models import Sequential
from keras.layers import Dense
```
构建模型，构建一个序列模型，输入维度为100，添加一个包含64个单元的全连接层，再添加只有一个单元的输出层
``` python
model = Sequential()
model.add(Dense(units=64,activation='relu',input_dim=100))
model.add(Dense(units=1,activation='sigmoid'))
```
模型编译，设定二元交叉熵损失函数和随机梯度下降优化器，另加一个准确率为评估指标
``` python
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
```
模型训练，评估和预测
``` python
model.fit(train_data,train_label,epochs=5,batch_size=32)
loss_and_metrics = model.evaluate(train_data, train_label, batch_size=128)
classes = model.predict(test_data,batch_size=128)
```
