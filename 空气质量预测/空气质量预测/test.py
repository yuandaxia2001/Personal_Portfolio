import keras_tuner
import numpy as np
import pandas as pd
import 数据综合 as data_merge
from 模型文件 import get_single_model,get_merge_model
import random
import os
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,mean_squared_error,r2_score

import matplotlib.pyplot as plt
def plots(x,y,model_name,polution_name,time):
    if model_name=='merge merge':
        model_name='merge all'
    fig,ax=plt.subplots()
    plt.xlim(0,250)
    plt.ylim(0,250)
    liner=np.arange(0,250)
    ax.plot(liner,liner,color='red')
    ax.scatter(x,y,color='blue',s=0.5)
    plt.xlabel('Observed value (μg/m³)')
    plt.ylabel('Predicted value (μg/m³)')
    plt.title(f'{model_name}')
    plt.savefig(f'图片/{polution_name}_{time}_{model_name}.jpg', dpi=1000)
    plt.show()



def evaluate(model,x_test,y_test,is_plt=False,model_name=None,polution_name=None,time=None):
    #评估模型的训练效果
    y_predict = model.predict(x_test)
    mape = mean_absolute_percentage_error(y_test, y_predict) * 100
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predict)

    print(f'{mae:.4f}  {mape:.4f}%  {mse:.4f}  {rmse:.4f}  {r2:.4f}')
    if is_plt:
        plots(y_test,y_predict,model_name,polution_name,time)

def load_dataset(data_type):
    '''
    :param data_type: 要加载的数据集的种类,train,test,valid中的一个
    :return: 加载好的数据集
    '''
    X=np.load(f'data/x_{data_type}.npz',allow_pickle=True)
    Y=np.load(f'data/y_{data_type}.npz',allow_pickle=True)
    X=X['arr_0']
    Y=Y['arr_0']
    X=X.item()
    Y=Y.item()
    return X,Y


if __name__=='__main__':
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

    X_train,Y_train=load_dataset('train')
    X_valid,Y_valid=load_dataset('valid')
    X_test,Y_test=load_dataset('test')


    steps=[1,2,4,6,8,12,18,24]

    model_types = ['tcn', 'simple_rnn', 'gru', 'lstm']
    monitor = EarlyStopping(patience=10, restore_best_weights=True)

    for step in steps:
        x_train, y_train = np.array(X_train[f'predict_{step}hours']), np.array(Y_train[f'predict_{step}hours'])
        x_valid, y_valid = np.array(X_valid[f'predict_{step}hours']), np.array(Y_valid[f'predict_{step}hours'])
        x_test, y_test = np.array(X_test[f'predict_{step}hours']), np.array(Y_test[f'predict_{step}hours'])

        best_models = {}  # 装有基模型数组字典，每个基模型数组将会被融合为一个模型

        # 训练基模型
        for model_type in model_types:
            best_model = get_single_model(model_type=model_type, model_num=4, x_train=x_train, y_train=y_train,
                                          x_valid=x_valid, y_valid=y_valid, monitor=monitor, batch_size=512,
                                          max_epochs=50,
                                          factor=8, iterations=3, seed=seed, directory=f'日志文件/{step}hours_test',
                                          overwrite=True)
            best_models[model_type] = best_model

            print(f'======={model_type}单模型{step}hours的前4名模型预测情况=======')
            print('mae  mape  mse  rmse  r2')
            for i, model in enumerate(best_model):
                evaluate(model, x_train, y_train, is_plt=False)

            print(f'======={model_type}单模型{step}hours的前4名模型预测情况=======')
            print('mae  mape  mse  rmse  r2')
            for i, model in enumerate(best_model):
                evaluate(model, x_valid, y_valid, is_plt=False)

            print(f'======={model_type}单模型{step}hours的前4名模型预测情况=======')
            print('mae  mape  mse  rmse  r2')
            for i, model in enumerate(best_model):
                evaluate(model, x_test, y_test, is_plt=False)
            break

        # 训练融合模型
        merge_models = []
        # for models in best_models.values():
        #     merge_models.append(models[0])
        # best_models['merge'] = merge_models

        for name, models in best_models.items():
            merge_model = get_merge_model(models, model_name=name, x_train=x_train, y_train=y_train, x_valid=x_valid,
                                          y_valid=y_valid, monitor=monitor, batch_size=512, max_epochs=50, factor=8,
                                          iterations=1, seed=seed,
                                          directory=f'日志文件/{step}hours', overwrite=True)

            print(f'======={name}融合模型模型{step}hours预测情况=======')
            print('mae  mape  mse  rmse  r2')
            for model in merge_model:
                evaluate(model, [x_train] * len(models), y_train, is_plt=False, model_name='merge ' + name,
                     time=step)

            print(f'======={name}融合模型模型{step}hours预测情况=======')
            print('mae  mape  mse  rmse  r2')
            for model in merge_model:
                evaluate(model, [x_valid] * len(models), y_valid, is_plt=False, model_name='merge ' + name,
                     time=step)


            print(f'======={name}融合模型模型{step}hours预测情况=======')
            print('mae  mape  mse  rmse  r2')
            for model in merge_model:
                evaluate(model, [x_test] * len(models), y_test, is_plt=False, model_name='merge ' + name,
                    time=step)





























