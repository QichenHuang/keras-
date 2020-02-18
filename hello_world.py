from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 构建Sequential模型
model = Sequential()
# 向Sequential模型中添加两个全连接层
model.add(Dense(units=64,activation='relu',input_dim=100))
model.add(Dense(units=1,activation='sigmoid'))
# 对模型进行编译，设定损失函数和优化器等
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
# 用随机数生成训练数据,标签和测试数据
train_data = np.random.random((1000,100))
train_label = np.random.randint(2,size=1000)
test_data = np.random.random((1000,100))
# 模型训练
model.fit(train_data,train_label,epochs=5,batch_size=32)
# 模型评估
loss_and_metrics = model.evaluate(train_data, train_label, batch_size=128)
# 类别预测
classes = model.predict(test_data,batch_size=128)



