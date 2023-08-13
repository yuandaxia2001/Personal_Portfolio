from keras.models import Model
from keras.layers import Input,Conv1D,LSTM,Dropout,GRU,SimpleRNN,Dense,BatchNormalization
from tensorflow.keras.optimizers import Adam
import keras

class tcn:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_tcn(self,hp):
        num_layers=hp.Int('num_layers',min_value=1,max_value=3)

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=Conv1D(filters=hp.Int(f'units_{i+1}',min_value=64,max_value=256,step=64),
                     kernel_size=2,strides=1,padding='causal',activation='relu',
                     dilation_rate=2**i)(x)

        x = Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=2**num_layers)(x)

        x = x[:, -1, :]
        outputs=Dense(units=1)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log')),
                      loss='mse',metrics='mae')
        return model



class lstm:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_lstm(self,hp):
        num_layers=hp.Int('num_layers',min_value=1,max_value=3)

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=LSTM(units=hp.Int(f'units_{i+1}',min_value=64,max_value=256,step=64),
                   activation='relu',return_sequences=True)(x)

        x=LSTM(units=64,activation='relu',return_sequences=False)(x)

        outputs=Dense(units=1,activation=None)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log')),
                      loss='mse',metrics='mae')
        return model

class gru:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_gru(self,hp):
        num_layers=hp.Int('num_layers',min_value=1,max_value=3)

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=GRU(units=hp.Int(f'units_{i+1}',min_value=64,max_value=256,step=64),
                  activation='relu',return_sequences=True)(x)

        x=GRU(units=64,activation='relu',return_sequences=False)(x)
        outputs=Dense(units=1,activation=None)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log')),
                      loss='mse',metrics='mae')
        return model


class simple_rnn:
    def __init__(self,input_shape):
        self.input_shape=input_shape

    def build_simple_rnn(self,hp):
        num_layers=hp.Int('num_layers',min_value=1,max_value=3)

        inputs=Input(shape=self.input_shape)
        x=BatchNormalization()(inputs)
        for i in range(num_layers):
            x=SimpleRNN(units=hp.Int(f'units_{i+1}',min_value=64,max_value=256,step=64),
                  activation='relu',return_sequences=True)(x)

        x=SimpleRNN(units=64,activation='relu',return_sequences=False)(x)
        outputs=Dense(units=1,activation=None)(x)

        model=Model(inputs,outputs)
        model.compile(optimizer=Adam(hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log')),
                      loss='mse',metrics='mae')
        return model


class merge_model:#model_list为一维列表，装有所有要融合的基模型
    def __init__(self,model_list):
        self.model_list=model_list

    def build_merge_model(self,hp):
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

        x = Dense(units=hp.Choice(f'dense{i + 1}', [256, 512, 1024, 2048, 4096]),
                  name=f'dense{i + 1}', activation='relu')(x)
        x = Dropout(hp.Float(f'drop{i + 1}', min_value=0.1, max_value=0.4, step=0.1),
                    name=f'dropout_{i + 1}')(x)


        out = Dense(units=1,name='dense_out')(x)

        model = Model(inputs=input_list, outputs=out)
        model.compile(Adam(hp.Float('lr',min_value=1e-4,max_value=1e-2,sampling='log')),
                      loss='mse',metrics='mae')
        for layer in model.layers:
            layer.trainable = False

        for i in range(2):
            model.layers[-1-2*i].trainable = True

        return model



from keras_tuner import Hyperband

def get_single_model(model_type,model_num,x_train,y_train,x_valid,y_valid,
              monitor,batch_size,max_epochs,factor,iterations,seed,directory,overwrite,which_one):
    '''
    获取单一种类的模型
    :param model_type:要获取的模型的种类
    :param model_num: 获取模型的数目
    :param x_train: 训练集
    :param y_train: 训练集
    :param x_valid: 验证集
    :param y_valid: 验证集
    :param monitor: 训练模型的监视器
    :param batch_size: 训练模型时的批尺寸
    :param max_epochs: 最大训练轮数
    :param factor: 递减因子
    :param iterations: 寻参轮数
    :param seed: 随机种子
    :param directory: 日志文件存放的目录
    :param overwrite: 是否重新写日志文件
    :param whichone:这是第几个模型，用于命名
    :return: 返回装有model_num个数的种类为model_type的模型数组
    '''
    all_models={'tcn':tcn(input_shape=x_train.shape[1:]).build_tcn,
                'lstm':lstm(input_shape=x_train.shape[1:]).build_lstm,
                'gru':gru(input_shape=x_train.shape[1:]).build_gru,
                'simple_rnn':simple_rnn(input_shape=x_train.shape[1:]).build_simple_rnn}

    hypermodel=all_models[model_type]

    tuner = Hyperband(hypermodel, objective='val_loss', max_epochs=max_epochs,
                      factor=factor, seed=seed, directory=directory, hyperband_iterations=iterations,
                      project_name='单模型/' + model_type + f'/the {which_one}th', overwrite=overwrite)
    tuner.search(x_train, y_train, batch_size=batch_size, callbacks=[monitor],
                 validation_data=(x_valid, y_valid), verbose=0)

    best_models=tuner.get_best_models(model_num)
    return best_models[0]

def get_merge_model(model_list,model_name,x_train,y_train,x_valid,y_valid,monitor,
              model_num,batch_size,max_epochs,factor,iterations,seed,directory,overwrite):
    '''
    获得融合后的模型
    :param model_list:要融合的模型的数组
    :param model_name: 要融合的模型的名称，用于日志文件的命名
    :param x_train: 训练集
    :param y_train: 训练集
    :param x_valid: 验证集
    :param y_valid: 验证集
    :param monitor: 监视器
    :param batch_size: 批尺寸
    :param max_epochs: 最大训练轮数
    :param factor: 递减因子
    :param iterations: 寻参轮数
    :param seed: 随机种子
    :param directory: 日志文件路径
    :param overwrite: 是否重新写日志文件
    :return: 一个训练好的融合模型，融合了model_list中的所有基模型
    '''
    hypermodel=merge_model(model_list).build_merge_model

    tuner=Hyperband(hypermodel, objective='val_loss', max_epochs=max_epochs,
                      factor=factor, seed=seed, directory=directory, hyperband_iterations=iterations,
                      project_name='融合模型/' + model_name, overwrite=overwrite)
    tuner.search(x_train, y_train, batch_size=batch_size, callbacks=[monitor],
                 validation_data=(x_valid, y_valid), verbose=0)
    return tuner.get_best_models(model_num)

