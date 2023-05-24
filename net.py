# 设置LSTM块的数量 n_block^2
n_blocks = 8
# 定义输入层
inputs = Input(shape=(64, 64, 3))

# 第一个卷积层 输出：64x64x16
conv1 = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(inputs)

# 第二个卷积层 same填充 输出大小为：64x64x1  只进行相应元素的相加，H,W,C都不改变
conv2 = Add()([Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(conv1) for _ in range(16)])

# 将特征图分为64个块，每个块大小为8x8，输出大小为：64x8x8x1
conv2 = Reshape(n_blocks*n_blocks, 8, 8, 1)(conv2)

# 定义 LSTM 层 输入为64x8x8x1
lstm_layer = LSTM(64, input_shape=(None, 8, 8, 1), return_sequences=True)(conv2)
lstm_layer = LSTM(64, return_sequences=True)(lstm_layer)
lstm_layer = LSTM(64)(lstm_layer)

# 将 LSTM 层输出的特征向量转换为 16x16 的块   输出大小为16x16x64
block_layer = Dense(16*16*64, activation='softmax')(lstm_layer)   # 全连接
block_layer = Reshape((16, 16, 64))(block_layer)

# 分离出每个通道并转换为大小为(64, 64, 1)的张量
channel_1 = Lambda(lambda x: x[:, :, :, 0])(block_layer)
channel_1 = Reshape((64, 64, 1))(channel_1)
channel_2 = Lambda(lambda x: x[:, :, :, 1])(block_layer)
channel_2 = Reshape((64, 64, 1))(channel_2)
channel_3 = Lambda(lambda x: x[:, :, :, 2])(block_layer)
channel_3 = Reshape((64, 64, 1))(channel_3)
channel_4 = Lambda(lambda x: x[:, :, :, 3])(block_layer)
channel_4 = Reshape((64, 64, 1))(channel_4)

# 拼接四个通道成为大小为(128, 128, 1)的张量
merged_layer = Concatenate(axis=1)([channel_1, channel_2, channel_3, channel_4])
merged_layer = Concatenate(axis=2)([merged_layer]*2)

# 定义三个卷积层和一个最大池化层 输出分别为 128x128x32、64x64x32、64x64x2、64x64x2
conv3 = Conv2D(32, (3, 3), padding='same', activation='relu')(merged_layer)
pool = MaxPooling2D((2, 2))(conv3)
conv4 = Conv2D(2, (3, 3), padding='same', activation='relu')(pool)
conv5 = Conv2D(2, (3, 3), padding='same', activation='softmax')(conv4)

# 得到 64x64x2 的置信图
confidence_map = conv5

# 定义输出层
output = Reshape(target_shape=(64, 64))(confidence_map)
# 定义模型
model = Model(inputs=inputs, outputs=output)
