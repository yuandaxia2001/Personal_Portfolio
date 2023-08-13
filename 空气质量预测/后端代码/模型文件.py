from keras.models import Model
from keras.layers import Input,Conv1D,LSTM,Dropout,GRU,SimpleRNN,Dense,BatchNormalization,Activation,Attention
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras
import numpy as np
import random
import os
import tensorflow as tf

seed = 42
random.seed(seed)  # 为python设置随机种子
os.environ['PYTHONHASHSEED'] = str(seed)  # tf gpu fix seed, please `pip install tensorflow-determinism` first
os.environ['TF_DETERMINISTIC_OPS'] = str(1)
os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
np.random.seed(seed)  # 为numpy设置随机种子
tf.random.set_seed(seed)  # tf cpu fix seed

# 获取所有 GPU 设备列表
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置 GPU 显存占用为按需分配，增长式
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:

        print(e)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽警告信息
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

predict_hour=1
unit=64
num_layers=0

class tcn:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_tcn(self):

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=Conv1D(filters=unit,
                     kernel_size=2,strides=1,padding='causal',activation='relu',
                     dilation_rate=2**i)(x)
            x=BatchNormalization()(x)

        x = Conv1D(filters=unit, kernel_size=2, padding='causal', dilation_rate=2**num_layers)(x)

        x = x[:, -1, :]

        outputs=Dense(units=predict_hour)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(0.005),
                      loss='mse',metrics='mae')
        return model

class lstm:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_lstm(self):

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=LSTM(units=unit,
                   activation='relu',return_sequences=True)(x)
            x = BatchNormalization()(x)

        x=LSTM(units=unit,activation='relu',return_sequences=False)(x)

        outputs=Dense(units=predict_hour,activation=None)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(),
                      loss='mse',metrics='mae')
        return model

class gru:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_gru(self):

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=GRU(units=unit,
                  activation='relu',return_sequences=True)(x)
            x = BatchNormalization()(x)

        x=GRU(units=unit,activation='relu',return_sequences=False)(x)
        outputs=Dense(units=predict_hour,activation=None)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(),
                      loss='mse',metrics='mae')
        return model

class simple_rnn:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_simple_rnn(self):

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=SimpleRNN(units=unit,
                  activation='relu',return_sequences=True)(x)
            x = BatchNormalization()(x)

        x=SimpleRNN(units=unit,activation='relu',return_sequences=False)(x)
        outputs=Dense(units=predict_hour,activation=None)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(),
                      loss='mse',metrics='mae')
        return model

class merge_model:#model_list为一维列表，装有所有要融合的基模型
    def __init__(self,model_list):
        self.model_list=model_list

    def build_merge_model(self):
        input_list = []
        output_list = []
        for i, item in enumerate(self.model_list):
            for layer in item.layers:
                layer._name = layer.name + f'_{i + 1}'
            temp = item
            inputs = temp.input
            input_list.append(inputs)
            x = temp.layers[-2].output
            output_list.append(x)

        x = keras.layers.Concatenate()(output_list)
        x = Dense(units=512,
                  name=f'dense{i + 1}', activation='relu')(x)
        x = Dropout(0.1,
                    name=f'dropout_{i + 1}')(x)


        out = Dense(units=1,name='dense_out')(x)

        model = Model(inputs=input_list, outputs=out)
        model.compile(Adam(),
                      loss='mse',metrics='mae')
        for layer in model.layers:
            layer.trainable = True

        # for layer in model.layers:
        #     layer.trainable = False
        #
        # for i in range(2):
        #     model.layers[-1-2*i].trainable = True

        return model
