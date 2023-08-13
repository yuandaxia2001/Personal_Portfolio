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

def get_random_feature(feature_num,all_feature):
    '''
    :param feature_num:需要随机选取的特征数目
    :return: 返回一个有100个一维数组的数组，实际上用不到100个基模型，但是不嫌多
    '''
    result=[]
    for i in range(100):
        x=np.random.choice(all_feature-1,feature_num-1,replace=False)
        x=x+1
        x=np.append(x,0) # 随机选择feature_num个特征，其中一定包括pm2.5
        x=np.sort(x)
        result.append(x)
    return result


def train_basemodel(model_type,model_num,feature_choice,X_train,Y_train,X_valid,Y_valid,X_test,Y_test,monitor,batch_size,
                    max_epochs,factor,iterations,seed,directory,overwrite):
    '''

    :param model_type: 要生成的基模型种类
    :param model_num: 基模型的个数
    :return:一个列表，其中每个元素是元组，元组中有：模型，训练集，验证，测试集
    '''
    result=[]
    print(f'======={model_type}单模型{step}hours{model_num}个模型预测情况=======')
    print('mae  mape  mse  rmse  r2')
    for i in range(model_num):
        x=feature_choice[i]
        x_train = X_train[:, :, x]
        y_train = Y_train
        x_valid = X_valid[:, :, x]
        y_valid = Y_valid
        x_test = X_test[:, :, x]
        y_test = Y_test
        best_model = get_single_model(model_type=model_type, model_num=1, x_train=x_train, y_train=y_train,
                                      x_valid=x_valid, y_valid=y_valid, monitor=monitor, batch_size=batch_size,
                                      max_epochs=max_epochs, factor=factor, iterations=iterations,
                                      seed=seed+i, directory=directory, overwrite=overwrite,
                                      which_one=i+1) #保证寻参时随机获得的参数不同，所以seed为seed+i
        result.append((best_model,x_train,y_train,x_valid,y_valid,x_test,y_test))


        evaluate(best_model, x_test, y_test, is_plt=False)

    print('')
    return result






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


    steps=[1,6,18,24] #[1,2,4,6,8,12,18,24]
    model_types = ['tcn', 'simple_rnn', 'lstm', 'gru']
    feature_nums = [8, 16, 24]  # [4,8,12,16,20,24]
    all_feature=24

    random_feature={}
    for step in steps:
        for model_type in model_types:
            for feature_num in feature_nums:
                random_feature[f'{step}_{model_type}_{feature_num}']=get_random_feature(
                    feature_num,all_feature)



    monitor = EarlyStopping(patience=8, restore_best_weights=True)

    for step in steps:
        x_train, y_train = np.array(X_train[f'predict_{step}hours']), np.array(Y_train[f'predict_{step}hours'])
        x_valid, y_valid = np.array(X_valid[f'predict_{step}hours']), np.array(Y_valid[f'predict_{step}hours'])
        x_test, y_test = np.array(X_test[f'predict_{step}hours']), np.array(Y_test[f'predict_{step}hours'])

        base_models = {}  # 装有基模型数组字典，每个基模型数组将会被融合为一个模型
        # 训练基模型
        for model_type in model_types:
            for feature_num in feature_nums:
                temp=train_basemodel(model_type=model_type,feature_choice=random_feature[f'{step}_{model_type}_'
                                                f'{feature_num}'],model_num=4,X_train=x_train,
                                Y_train=y_train,X_valid=x_valid,Y_valid=y_valid,X_test=x_test,Y_test=y_test,
                                monitor=monitor,batch_size=512,max_epochs=50,factor=8,iterations=1,
                                     seed=seed,directory=f'日志文件1/{step}hours/{feature_num} features',overwrite=False)


                merge_x_train = []
                merge_y_train = []
                merge_x_valid = []
                merge_y_valid = []
                merge_x_test = []
                merge_y_test = []
                merge_basemodel = []
                merge_tuple=(merge_basemodel,merge_x_train,merge_y_train,merge_x_valid,merge_y_valid,
                             merge_x_test,merge_y_test)

                for item in temp:
                    for i in range(7):
                        merge_tuple[i].append(item[i])
                merge_y_train=merge_y_train[0]
                merge_y_test=merge_y_test[0]
                merge_y_valid=merge_y_valid[0]

                base_models[f'{model_type}_{feature_num}features'] = merge_basemodel # 保存基模型，用于多模型融合

                merge_single_model=get_merge_model(merge_basemodel,model_type,merge_x_train,merge_y_train,
                                                   merge_x_valid,merge_y_valid,monitor,model_num=5,
                                                   batch_size=512,max_epochs=50,factor=8,iterations=1,
                                                   seed=seed,directory=f'日志文件1/{step}hours/{feature_num} features',
                                                   overwrite=False)

                print(f'======={model_type}融合模型模型{step}hours预测情况=======')
                print('mae  mape  mse  rmse  r2')
                for model in merge_single_model:
                    evaluate(model,merge_x_test,merge_y_test)


        # # 训练融合模型
        # for model_type in model_types:
        #     for feature_num in feature_nums:































