# 模型的命名约定：污染物名称_预测的时长_模型种类.h5
# 比如：PM2.5_6hour_tcn.h5

import os
import pandas as pd
import numpy as np
import keras
import 模型文件 as model_file
from sklearn.model_selection import train_test_split
import sys

def get_model_info(pollution_type, predict_time, model_type, model_road):  # ok
    '''
    获取某个模型的信息
    :param pollution_type:污染物种类
    :param predict_time:预测的时长
    :param model_type:模型的种类
    :param model_road:存模型的文件夹路径
    :return:如果这个模型存在，返回True，如果这个模型不存在，返回False
    '''
    file_names = os.listdir(model_road)
    if f"{pollution_type}_{predict_time}hour_{model_type}.h5" in file_names:
        return True
    else:
        return False


def read_dataset_head(dataset_road):  # ok
    '''
    读取数据集的表头（数据集是一个.csv文件）
    :param dataset_road:数据集的路径
    :return:返回一个列表，即数据集的表头，例如：[“PM2.5”,”AQI”,”PM10”,”S02”]
    '''
    res = pd.read_csv(dataset_road)
    return res.columns.to_list()


def get_predict(pollution_type, model_type, predict_time, model_road, dataset_road):
    '''
    获得一个预测值
    :param pollution_type:污染物类别
    :param model_type:模型种类
    :param predict_time:预测时长
    :param model_road:模型存放的文件夹路径
    :param dataset_road: 数据集的路径
    :return:一个浮点数，该污染物的预测值，如果没有这个模型，则返回-1
    '''
    if get_model_info(pollution_type, predict_time, model_type, model_road):
        model = keras.models.load_model(model_road + f"/{pollution_type}_{predict_time}hour_{model_type}.h5")
        dataset = pd.read_csv(dataset_road)
        x = np.array(dataset[0:24])
        x = np.array(x)
        x = np.expand_dims(x, axis=0)
        if (model_type == 'hybrid'):
            x = [x] * len(model.inputs)

        return model.predict(x)[0][0]
    else:
        return -1


base_models = {}
model_value = {}

from keras.callbacks import EarlyStopping

monitor = EarlyStopping(patience=15)


def train_model(pollution_type, model_type, predict_time, model_road, dataset_road):  # ok
    '''
    训练出一个模型，可以选择污染物，预测时长，模型种类等细粒度的参数
    :param pollution_type:污染物类别
    :param model_type:模型种类
    :param predict_time:预测时长
    :param model_road:模型存放的文件夹路径
    :return:
    '''
    dir = os.listdir(model_road)
    if 'x_train.npy' not in dir:
        process_data(dataset_road, model_road)

    x_train = np.load(model_road + '/x_train.npy')
    x_test = np.load(model_road + '/x_test.npy')
    y_train = np.load(model_road + '/y_train.npy')
    y_test = np.load(model_road + '/y_test.npy')
    dataset_head=read_dataset_head(dataset_road)
    pollution_index = None
    for i, pollution in enumerate(dataset_head):  # 获得污染物在y中的索引
        if pollution == pollution_type:
            pollution_index = i
            break

    if x_train.shape[0] < 1000:
        monitor = EarlyStopping(patience=50, min_delta=5, restore_best_weights=True)
        epochs = 500
        if model_type == 'lstm' or model_type == 'rnn':
            monitor = EarlyStopping(patience=100, min_delta=5, restore_best_weights=True)

    elif x_train.shape[0] < 2000:
        monitor = EarlyStopping(patience=50, min_delta=4, restore_best_weights=True)
        epochs = 400
        if model_type == 'lstm' or model_type == 'rnn':
            monitor = EarlyStopping(patience=80, min_delta=4, restore_best_weights=True)

    elif x_train.shape[0] < 5000:
        monitor = EarlyStopping(patience=40, min_delta=3, restore_best_weights=True)
        epochs = 200
        if model_type == 'lstm' or model_type == 'rnn':
            monitor = EarlyStopping(patience=60, min_delta=3, restore_best_weights=True)

    elif x_train.shape[0] < 10000:
        monitor = EarlyStopping(patience=30, min_delta=2, restore_best_weights=True)
        epochs = 100

    else:
        monitor = EarlyStopping(patience=20, min_delta=1, restore_best_weights=True)
        epochs = 60

    if model_type == 'hybrid':
        base_models = {}
        base_models[pollution_type] = []

        model_types=['tcn', 'rnn', 'lstm', 'gru']

        for model_name in model_types:
            if f'{pollution_type}_{predict_time}hour_{model_name}.h5' in dir:
                base_models[pollution_type].append(keras.models.load_model(
                    model_road + f'/{pollution_type}_{predict_time}hour_{model_name}.h5'
                ))

        if pollution_type in base_models:
            model = model_file.merge_model(base_models[pollution_type]).build_merge_model()
            model.fit(x=[x_train] * len(base_models[pollution_type]), y=y_train[predict_time - 1, :, pollution_index],
                      batch_size=256, epochs=epochs, callbacks=[monitor], verbose=0,
                      validation_data=([x_test] * len(base_models[pollution_type]),
                                       y_test[predict_time - 1, :, pollution_index]))

            model.save(model_road + f'/{pollution_type}_{predict_time}hour_{model_type}.h5')
            result = evaluate(model, [x_test] * len(base_models[pollution_type]),
                              y_test[predict_time - 1, :, pollution_index])

    else:
        model_dir = {'tcn': model_file.tcn(x_train.shape[1:]).build_tcn(),
                     'rnn': model_file.simple_rnn(x_train.shape[1:]).build_simple_rnn(),
                     'lstm': model_file.lstm(x_train.shape[1:]).build_lstm(),
                     'gru': model_file.gru(x_train.shape[1:]).build_gru()}
        model = model_dir[model_type]
        model.fit(x=x_train, y=y_train[predict_time - 1, :, pollution_index], batch_size=256, epochs=epochs,
                  callbacks=[monitor],
                  verbose=0, validation_data=(x_test, y_test[predict_time - 1, :, pollution_index]))

        model.save(model_road + f'/{pollution_type}_{predict_time}hour_{model_type}.h5')

        result = evaluate(model, x_test, y_test[predict_time - 1, :, pollution_index])

    f = open(model_road + '/模型评估数据.txt', 'a')
    f.write(f'{pollution_type}_{predict_time - 1}hour_{model_type}\n')
    for item in result:
        f.write(f'{item:.4f}  ')
    f.write('\n')
    f.close()
    global model_value
    model_value.clear()

    return result


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, x_test, y_test):  # ok
    # 评估模型的训练效果
    y_predict = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predict)
    return (mae, rmse, r2)


def train_one_pollution_model(pollution_type, model_road):
    '''
    训练出预测某一个污染物的所有模型，由于有5种模型，可以预测6个小时，所以一次性会训练出30个模型，
    这个函数的缺点是不能让前端知道一个模型是否已经训练完了，只有训练完30个模型才能有返回值，
    但是前端可以参考这个函数的实现，而不直接调用这个函数
    :param pollution_type: 污染物种类
    :param model_road: 存放模型的文件夹路径
    :param dataset_road: 数据集路径
    :return:
    '''
    model_types = ['tcn', 'rnn', 'lstm', 'gru', 'hybrid']
    for i in range(0, 6):
        for model_type in model_types:
            train_model(pollution_type, model_type, predict_time=i + 1, model_road=model_road)




dataset_head = []


def process_data(dataset_road, model_road):  # ok
    '''
    数据集预处理,训练集放在x中,标签放在y中
    :param dataset_road:数据集路径
    :param model_road:模型文件夹路径
    :return:
    '''
    global dataset_head, x_train, x_test, y_train, y_test, base_models

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    dataset_head = []
    x = []
    y = []
    base_models = {}
    # 初始化一次所有变量

    res = pd.read_csv(dataset_road)
    dataset_head = res.columns.to_list()
    for item in dataset_head:
        base_models[item] = []

    for i in range(6):
        y.append([])  # 初始化y

    for i in range(0, len(res) - 24 - 6):
        x.append(np.array(res[i:i + 24]))
        for j in range(6):
            y[j].append(np.array(res[i + 24 + j + 1:i + 24 + j + 1 + 1]))  # 获得6h标签

    for i in range(6):
        x_train, x_test, y_train_temp, y_test_temp = train_test_split(x, y[i], test_size=0.2, random_state=42)
        y_train.append(y_train_temp)
        y_test.append(y_test_temp)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[3]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[3]))

    np.save(model_road + '/x_train.npy', x_train)
    np.save(model_road + '/x_test.npy', x_test)
    np.save(model_road + '/y_train.npy', y_train)
    np.save(model_road + '/y_test.npy', y_test)


def get_model_value(pollution_type, predict_time, model_type, model_road):
    '''
    获取某个模型的评估指标
    :param pollution_type:污染物种类，字符串
    :param predict_time: 预测时长，整数
    :param model_type: 模型种类，tcn，rnn之类的，小写
    :param model_road: 模型文件夹的路径，后端默认模型评估数据文件会存在这个路径，因为后端就是把这个文件保存在模型文件夹中的
    :return:一个元组，有3个数，从左至右依次是mae，rmse，r²，如果中途失败，返回-1
    '''
    global model_value
    if model_value == {}:
        try:
            f = open(model_road + '/模型评估数据.txt', 'r')
        except:
            return -1

        while (True):
            model_name = f.readline()
            if model_name == '':
                break
            else:
                temp = tuple(map(float, f.readline().split()))
                model_value[model_name] = temp
        f.close()

    if f"{pollution_type}_{predict_time - 1}hour_{model_type}\n" in model_value:
        return model_value[f'{pollution_type}_{predict_time - 1}hour_{model_type}\n']
    else:
        return -1

# process_data("C:/Users/yuan/Desktop/UI设计初版/UI设计/data/长沙空气质量汇总 - 副本.csv")
#
# print(train_model('PM2.5','tcn',1,'C:/Users/yuan/Desktop/model'))
# print(train_model('PM2.5','rnn',1,'C:/Users/yuan/Desktop/model'))
# print(train_model('PM2.5','lstm',1,'C:/Users/yuan/Desktop/model'))
# print(train_model('PM2.5','gru',1,'C:/Users/yuan/Desktop/model'))
# print(train_model('PM2.5','hybrid',1,'C:/Users/yuan/Desktop/model'))

# print(get_predict('PM2.5','tcn',1,'C:/Users/yuan/Desktop/model',"C:/Users/yuan/Desktop/UI设计初版/UI设计/长沙空气质量汇总.csv"))

# print(process_data('C:/Users/yuan/Desktop/计算机省赛/每小时-训练集 - 副本.csv'))
# print(train_model('AQI','tcn',1,"C:/Users/yuan/Desktop/测试"))
# print(get_model_value('AQI',1,'tcn',"C:/Users/yuan/Desktop/测试"))

# print(train_model('PM2.5','tcn',6,'每日-模型文件夹','每日-训练集.csv'))

# print(train_model('PM2.5','rnn',6,'每日-模型文件夹','每日-训练集.csv'))

# print(train_model('PM2.5','hybrid',6,'每日-模型文件夹','每日-训练集.csv'))

# print(get_model_value('PM2.5',6,'hybrid','每日-模型文件夹'))

# print(get_predict('PM2.5','hybrid',6,'每日-模型文件夹','每日-训练集.csv'))


if __name__ == "__main__":

    # print(train_model('PM2.5','tcn',1, 'C:/Users/G/Desktop/Unity/AirPollutionForecast/Assets/Scripts/WeatherForecastModel/models/'))
    # print('finished')

    if sys.argv[1] == "read_dataset_head":
        print(read_dataset_head(sys.argv[2]))

    elif sys.argv[1] == "train_model":
        print(train_model(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]))
        
    elif sys.argv[1] == "get_predict":
        print(get_predict(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))
    
    elif sys.argv[1] == "get_model_info":
        print(get_model_info(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]))
    
    elif sys.argv[1] == "get_model_value":
        print( get_model_value(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]))

    
    else :
        print("Refer to an unexisted func!")










