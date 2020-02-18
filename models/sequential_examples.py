from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
# 构建模型
model1 = Sequential()
model1.add(Dense(32,activation='relu',input_shape=(100,)))
model1.add(Dense(10,activation='softmax'))

model2 = Sequential([
    Dense(32,activation='relu', input_shape=(100,)),
    Dense(10,activation='softmax')
])

model3 = Sequential([
    Dense(32,activation='relu', input_dim=100),
    Dense(10,activation='softmax')
])

model4 = Sequential()
model4.add(Dense(32,activation='relu', input_dim=100))
model4.add(Dense(10,activation='softmax'))
# 模型编译
model1.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# 模型训练
train_x = np.random.random((1000,100))
train_y = np.random.randint(10,size=(1000,1))
train_y_one_hot = to_categorical(train_y,num_classes=10)

history = model1.fit(train_x,train_y_one_hot,epochs=5,batch_size=32)
# 模型评估
evaluate_x = np.random.random((300,100))
evaluate_y = np.random.randint(10,size=(300,1))
evaluate_y_one_hot = to_categorical(evaluate_y,num_classes=10)

loss_and_metrics = model1.evaluate(evaluate_x,evaluate_y_one_hot,batch_size=32)
# 模型预测
test_x = np.random.random((300,100))

test_y = model1.predict(test_x,batch_size=32)
