from keras.layers import Input,Dense,concatenate
from keras.models import Model
# 线性结构的网络
input_1 = Input((16,))

output_1 = Dense(64, activation='relu')(input_1)
output_2 = Dense(64, activation='relu')(output_1)
output_3 = Dense(10, activation='softmax')(output_2)

model_1 = Model(inputs=input_1,outputs=output_3)
# 包含多种结构的网络
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
# 调用模型对象
output_9 = model_1(input_1)