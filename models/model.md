# Model  
前面提到过，Keras通过`keras.models.Model`类来组织各个网络层，通过将网络层线性叠加的`Sequential`属于`Model`的子类。
实际上，`Sequential`模型的编译、训练、评估和预测都是直接继承`Model`类的方法。

在正式介绍`Model`模型的构建之前，有几个概念需要理清：
- `Tensor`：张量，是模型中各个网络层之间传递流动的数据结构，在`keras.layers.Input`函数中提到，Keras的张量是对后端
（Theano，Tensorflow或CNTK）张量的封装，新增了两个属性：
	- `_keras_shape`：表示尺寸的元组
	- `_keras_history`：应用在当前张量的上一个网络层
- `Layer`：网络层，所有网络层的抽象基类。网络层对象是执行实际运算的模块，直接操作的数据对象是`Tensor`。
**注意，`Layer`类定义了`__call__`方法，这表明`Layer`对象是可调用的。**调用`Layer`对象需要提供输入张量，然后其返回输出张量。
- `Model`：模型，是组织各个网络层的类，可以看作是多个`Layer`对象的容器，网络结构必须用`Model`组织起来才能进行训练、
评估等操作。上一篇提到的`Sequential`类继承自`Model`类，是一个只能线性叠加网络层的容器。

## Model的构建
和`Sequential`对象的构建不同，`Model`对象的创建不接收网络层的对象，而是直接接受输入张量和输出张量。  
首先创建输入张量，使用`keras.layers.Input`函数创建输入张量
``` python
input_1 = Input((16,))
```
`Input`函数接受一个元组作为参数，表示不包含批次维度的数据尺寸，上面创建了尺寸为`(None,16)`的输入张量。

将输入张量作为参数直接调用网络层对象，可以得到输出张量。而一个网络层的输出张量可以作为另一个网络层的输入张量，
进而构建出整个网络结构
``` python
output_1 = Dense(64, activation='relu')(input_1)
output_2 = Dense(64, activation='relu')(output_1)
output_3 = Dense(10, activation='softmax')(output_2)
```
将整个网络的最初输入张量和最终输出张量作为创建`Model`对象的参数
``` python
model_1 = Model(inputs=input_1,outputs=output_3)
```
每一个keras的张量都保存了其作为哪些网络层的输入张量和输出张量。所以，给出输入和输出张量就可以溯源搜索到整个网络结构。

上面给出的例子仍然是单纯的将网络层线性的叠加得到的网络。通过直接调用网络层的方式，可以构建各种图结构的网络。如：
- 多个输入的网络，不同的输入通过各自的网络层后通过`keras.layers.concatenate`函数拼接成一个输出张量
- 多个输出的网络，除了通过`Input`函数创建的输入张量，其他张量都可以作为模型最终的输出张量在创建`Model`对象时传入。
这里的多个输出都会在总损失函数中占一定比重
- 共享网络层，多个张量都经过同一个网络层对象，则这些张量共享该网络层计算，得到各自的输出张量。
下面给出包含上述结构的例子：
``` python
# 三个输入张量
input_2 = Input((16,))
input_3 = Input((16,))
input_4 = Input((16,))

output_4 = Dense(32, activation='relu')(input_2)
# 共享网络层
shared_layer = Dense(64,activation='relu')
output_5 = shared_layer(input_3)
output_6 = shared_layer(input_4)
# 拼接多个张量
output_7 = concatenate([output_4,output_5,output_6])
# 最终输出张量
output_8 = Dense(10,activation='softmax')(output_7)
# 创建包含三个输入张量和两个输出张量的的模型
model_2 = Model(inputs=[input_2,input_3,input_4],outputs=[output_8,output_6])
```
## 深入Model
其实，和`Layer`对象一样，所有的`Model`对象都是可调用的。实际上，`Model`类继承自`Network`类，
对`Network`类增加了训练，评估等操作，而`Network`仅表示神经网络的拓扑结构，是以网络层为节点的有向无环图。
`Network`又是继承自`Layer`，也就是说，`Model`本身就是`Layer`的子类，可以当作一般的网络层来使用。
``` python
output_9 = model_1(input_1)
```

一个好的容器应该提供方便查看其状态的接口。下面对`Model`类的常用接口进行介绍。
- `layers`属性：返回构成模型的网络层列表
- `get_layer(name, index)`方法：基于名称或索引检索网络层对象，名称和索引同时提供时，优先考虑索引。
- `inputs`属性：返回模型的输入张量列表
- `outputs`属性：返回模型的输出属性列表
- `summary()`方法：打印出模型的总结表示
- `get_config()`方法：返回包含模型配置信息的字典
- `from_config(config)`类方法，接收调用`get_config()`得到的字典，创建相同配置的模型
- `get_weights()`方法，返回列表形式的模型中的所有权重，权重用Numpy array表示
- `set_weights(weights)`方法，设置模型中的权重值，权重参数为Numpy array的列表
- `to_json()`方法，生成json字符串，可以通过`keras.models.model_from_json(json_string)`方法将json字符串恢复成原模型，
字符串只包含结构，不包含权重
- `to_yaml()`方法，返回YAML字符串，可以通过`keras.models.model_from_yaml(yaml_string)`方法将YAML字符串恢复成原模型，
字符串只包含结构，不包含权重
- `save_weights(filepath)`方法，将模型权重保存在HDF5文件中
- `load_weights(filepath, by_name=False)`方法，从HDF5文件中加载模型权重；当加载的模型和原模型结构不同时，
将`by_name`参数设为`True`表示只加载相同名字的网络层  

当然，`Sequential`类是`Model`的子类，同样可以使用上述接口
## Model的编译
在上一篇对`Sequential`的介绍里，我们知道了`Sequential`的编译、训练、评估和预测的方法，从源代码的角度来看，
`Sequential`类本身并没有定义这几个方法，只是单纯的继承`Model`类的方法。下面具体介绍`Model`定义的`compile`方法。
``` python
compile(optimizer, loss=None, metrics=None, 
		loss_weights=None, sample_weight_mode=None, 
		weighted_metrics=None, target_tensors=None, **kwargs)
```
参数：
- optimizer：可以是预定义的优化器的字符串名字，也可以是Optimizer类的实例。
- loss：可以是预定义的损失函数的字符串名字，也可以是损失函数本身。当模型存在多个输出时，都使用该损失函数，
也可以对每个输出使用不同的损失函数，通过传入损失函数（或字符串名称）的列表，或者输出名称-损失函数（或字符串名称）的字典，
列表和字典的长度需等于输出数量
- metrics：当只有一个输出时，传入字符串指标名的列表或指标函数的列表；当存在多个输出时，可以传入列表或字典，如下：
	- 指标函数（或字符串名称）的列表：每个元素表示不同的评估指标，所有评估指标应用到所有输出中，如`['accuracy']`
	- 各输出指标的列表：每个元素表示对应输出的评估指标，列表长度等于输出数量，元素可以是字符串指标名，
	指标函数或列表，指标函数或列表，如`['accuracy',['accuracy','mse']]`
	- 各输出对应指标的字典：字典长度等于输出数量，每一个键值对对应一个输出，键为名称，值为字符串指标名，
	指标函数或列表，如`{'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`
- loss_weights：损失函数的权重，若指定该参数，可以传入列表或字典，长度等于输出数量；
列表的每个元素为标量，字典为名称到标量的映射。
- sample_weight_mode：给训练样本添加权重的模式。因为训练用的数据可能并非同等重要，有些样本是训练模型的关键，
有些样本微不足道几乎可以忽略，将所有样本一视同仁可能会影响模型效果。因此，给每一个样本赋予权值可以解决这个问题。
该参数需要配合[fit](#Model的训练)方法的`sample_weight`参数一起使用，取值只能为`None`或者`'temporal'`。
`None`表示为每一个样本提供权值，`'temporal'`表示为每一个样本的每一个时间步提供权值。具体细节请参考
`keras.engine.training_utils.standardize_weights`函数的源代码
- weighted_metrics：加权的评估指标列表。这里指定需要加权的指标的列表，配合[fit](#Model的训练)方法的`sample_weight`、
`class_weight`参数，或[evaluate](#Model的评估)方法的`sample_weight`参数，可以计算加权后的评估指标值。
- target_tensors：一般情况下，需要在训练时提供训练数据的标签，使用该参数可以提前指定训练数据的标签，
在使用`fit`函数时可以不用提供。
- \*\*kwargs：当使用Theano/CNTK后端时，这些参数会提供给`K.function`；当使用TensorFlow后端时，
这些参数提供给`tf.Session.run`

## Model的训练
Sequentail的训练方法`fit`也是直接继承自`Model`类，方法及参数如下：
``` python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
	validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
	sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, 
	validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False, **kwargs)
```
参数：
- `x`：样本数据，若只有一个输入，接受Numpy array；若有多个输入，可接受array的列表或从输入名称到arrays/tensors映射的字典
- `y`：样本标签，可接受的类型同`x`相同
- `batch_size`：训练的批次大小，默认为32
- `epochs`：一个`epochs`表示遍历一次所有给定的数据，这里指明要遍历多少次数据。注意，
当给出参数`initial_epoch`时，实际遍历的次数是`epochs - initial_epoch`
- `verbose`：啰嗦模式，可选0，1和2。0表示安静模式，不输出；1表示进度条，2表示每个epoch一行
- `callbacks`：`keras.callbacks.Callback`实例的列表，在训练和评估中应用的回调
- `validation_split`：0到1的浮点数，表示训练数据的多少部分作为验证数据。模型不会在验证数据上训练，
只会在每个epoch结束时计算损失函数和其他评估指标的值。划分验证数据会在打乱训练数据之前进行，
并将`x`和`y`最后的数据划分出来。
- `validation_data`：直接提供验证数据，该参数会使`validation_split`参数失效。接受Numpy arrays或tensors的元组，
`(x_val, y_val)或者(x_val, y_val, val_sample_weight)`。
- `shuffle`：是否打乱数据顺序，boolean值，或者字符串'batch'，表示处理HDF5数据的特殊选项
- `class_weight`：类别权重，从类别索引到权重值的字典，用于告诉模型在训练时更关注哪一类别的数据。该参数只在训练时使用，
具体细节请参考`keras.engine.training_utils.standardize_weights`函数的源代码
- `sample_weight`：样本权值，需要配合[compile](#Model的编译)函数的`sample_weight_mode`一起使用。
当`sample_weight_mode==None`时，该参数接受一维的权值array，长度等于样本数量。当`sample_weight_mode==temporal`时，
该参数接受2维的权值array，尺寸等于`(y.shape[0], y.shape[1])`。具体细节请参考
`keras.engine.training_utils.standardize_weights`函数的源代码
- `initial_epoch`：模型训练的开始epoch，总共训练`epochs - initial_epoch`次epoch
- `steps_per_epoch`：遍历一个epoch分多少步。
- `validation_steps`：在`steps_per_epoch`被指定了之后才有意义，表示验证的总步数
- `validation_freq`：验证的频率，可以是整数也可以是列表，元组等。如果是整数i，表示训练i个epoch后进行验证；
若为列表形式的[1,2,10]，表示在执行第1、2和10个epoch后进行验证
- `max_queue_size`：当输入x为生成器或者`keras.utils.Sequence`时才有意义，生成器队列的最大尺寸。默认为10
- `workers`：当输入x为生成器或者`keras.utils.Sequence`时才有意义，表示在使用基于进程的线程时的最大进程数。
- `use_multiprocessing`：布尔值，当输入x为生成器或者`keras.utils.Sequence`时才有意义，是否使用基于进程的线程
- `kwargs`：传递给后端的参数
## Model的评估
`Model`的评估使用`evaluate`函数，大部分参数作用和`fit`函数的参数相同
``` python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, 
	steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)
```
参数：
- `x`：评估样本数据，同[fit](#Model的训练)
- `y`：评估样本标签，同[fit](#Model的训练)
- `batch_size`：训练的批次大小，默认为32
- `verbose`：啰嗦模式，可选0，1。0表示安静模式，不输出；1表示进度条。
- `sample_weight`：样本权值，同[fit](#Model的训练)
- `steps`：评估的总步数
- `callbacks`：`keras.callbacks.Callback`实例的列表，评估时应用的回调
- `max_queue_size`：同[fit](#Model的训练)
- `workers`：同[fit](#Model的训练)
- `use_multiprocessing`：同[fit](#Model的训练)
## Model的预测
``` python
predict(x, batch_size=None, verbose=0, steps=None, callbacks=None, 
	max_queue_size=10, workers=1, use_multiprocessing=False)
```
参数：
- `x`：预测样本数据，同[fit](#Model的训练)
- `batch_size`：训练的批次大小，默认为32
- `verbose`：啰嗦模式，同[evaluate](#Model的评估)
- `steps`：评估的总步数
- `callbacks`：`keras.callbacks.Callback`实例的列表，预测时应用的回调
- `max_queue_size`：同[fit](#Model的训练)
- `workers`：同[fit](#Model的训练)
- `use_multiprocessing`：同[fit](#Model的训练)