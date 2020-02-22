# Sequential指南
在[第一个程序(Hello World)](../hello_world.md)中提到过，Keras通过对各种网络层的连接组合构建完整的神经网络，
而网络层的组合是通过`Model`类来完成的。所有网络结构中，最简单且最常见的组合方式就是网络层的线性叠加，
所以Keras提供了`Sequential`类，专门用以线性叠加多个网络层。`Sequential`实际上是`Model`的子类。

## Sequential的构建
和所有其他对象的创建一样，可以通过`Sequential()`来创建一个`Sequential`对象，然后通过`add`方法传入依次叠加的网络层
``` python
model1 = Sequential()
model1.add(Dense(32,activation='relu',input_shape=(100,)))
model1.add(Dense(10,activation='softmax'))
```
此外，也可以在创建`Sequential`对象的时候，直接传入网络层的列表，效果是一样的
``` python
model2 = Sequential([
    Dense(32,activation='relu', input_shape=(100,)),
    Dense(10,activation='softmax')
])
```
`Dense`创建对象的第一个参数表示神经元数量，而`input_shape`参数则是指定了输入数据的尺寸（不包含批次的维度）。  
实际上，当使用`Sequential`构建模型时，第一层网络必须指定输入数据的尺寸，其他后续网络层不需要指定输入数据的尺寸，
Keras内部逻辑可以自动求出。  
`input_shape`参数接收一个`tuple`，内部的元素可以是整数，也可以是`None`，`None`表示可以是任意整数。  
如上述代码，输入数据的尺寸应该是`(None,100)`，第一个维度`None`是批次的维度，表示可以是任意整数。  
当需要指定输入数据的批次大小时，可以在构建网络层对象的时候传递`batch_size`参数。

对于某些二维的网络层，如`Dense`，也可以通过指定`input_dim`参数指定输入数据的尺寸，`input_dim`参数可以接受一个整数。
上述的同等模型使用`input_dim`参数可以写作：
``` python
model3 = Sequential([
    Dense(32,activation='relu', input_dim=100),
    Dense(10,activation='softmax')
])

model4 = Sequential()
model4.add(Dense(32,activation='relu', input_dim=100))
model4.add(Dense(10,activation='softmax'))
```
## Sequential的编译
这里说的编译不是将源代码编译成目标程序的意思，而是对模型的训练过程进行配置或参数设定。  
编译使用模型的`compile`方法进行编译，配置内容通过三个参数传入。
- 优化器(optimizer)：可以是预定义的优化器的字符串名字，也可以是Optimizer类的实例。
- 损失函数(loss)：可以是预定义的损失函数的字符串名字，也可以是损失函数本身。
- 评估指标的列表(metrics)：可以是预定义的指标的字符串名字，也可以是指标函数本身。
对于分类任务通常设为`metrics=['accuracy']`  

其中，优化器和损失函数是必须的，评估指标是可选的
``` python
model1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
```
## Sequential的训练
Sequential模型的训练通过`fit`函数来实现。`fit`函数需要接受训练数据和标签，
训练数据和标签可以是`Numpy array`类型的数据，同时也可以设置`batch_size`等其他训练参数。  
函数返回一个`History`对象，该对象的`history`属性是一个字典，记录了损失函数和各项评估指标在不同周期的变化。
``` python
train_x = np.random.random((1000,100))
train_y = np.random.randint(10,size=(1000,1))
train_y_one_hot = to_categorical(train_y,num_classes=10)

history = model1.fit(train_x,train_y_one_hot,epochs=5,batch_size=32)
```
## Sequential的评估
Sequential模型的评估通过`evaluate`函数实现。`evaluate`函数接受`Numpy array`类型的评估数据和标签，
除了批次的维度可以不同，其他维度的尺寸和训练数据相同。其也可以接受`batch_size`等其他评估参数。  
函数返回一个列表，包含损失函数值和编译时指定的评估指标的值，
具体的指标顺序可通过model.metrics_names查看。
``` python
evaluate_x = np.random.random((300,100))
evaluate_y = np.random.randint(10,size=(300,1))
evaluate_y_one_hot = to_categorical(evaluate_y,num_classes=10)

loss_and_metrics = model1.evaluate(evaluate_x,evaluate_y_one_hot,batch_size=32)
```
## Sequential的预测
Sequential模型的预测通过`predict`函数实现。`predict`函数接受`Numpy array`类型的测试数据，
返回`Numpy array`类型的预测值。其也可以接受预测的其他预测参数。
``` python
test_x = np.random.random((300,100))

test_y = model1.predict(test_x,batch_size=32)
```
## 运行结果
``` python
>>>history.history
{'loss': [2.3427610492706297, 2.3128161392211912, 2.298412788391113, 2.2935817604064943, 2.278717025756836],
'accuracy': [0.104, 0.116, 0.124, 0.118, 0.133]}
>>>loss_and_metrics
[2.334255806605021, 0.10333333164453506]
>>>model.metrics_names
['loss', 'accuracy']
>>>classes
array([[0.17311491, 0.0711387 , 0.10607956, ..., 0.06384429, 0.08743901,
        0.07656132],
	   [0.19669338, 0.07826046, 0.11130829, ..., 0.07036302, 0.13512538,
        0.0727941 ],
       ...
       [0.14223056, 0.15673377, 0.06483066, ..., 0.08755407, 0.08570342,
        0.08559973],
       [0.12440495, 0.11137759, 0.08170168, ..., 0.08263049, 0.1177288 ,
        0.10077675]], dtype=float32)
>>>test_y.shape
(300,10)
```
需要完整代码的，请查看[源代码](./sequential_examples.py)  
[返回主页](../README.md)