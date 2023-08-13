import pandas as pd
import numpy as np


def check_data(data,timestep,max_error ,predict_step ):#检查当前数据是否两两之间相差的时间间隔不大于max_error
    for i in range(1,timestep + predict_step ):
        if data.iloc[i]['time']-data.iloc[i-1]['time']>max_error:
            return False
    return True

def get_data(timestep,predict_step=1,predict_type='PM2.5',max_error=pd.Timedelta('1 hours')):
    '''
    timestep:时间步长
    predict_step:预测未来第多少个小时的污染物浓度
    predict_type：污染物种类
    max_error：两组数据允许的最大时间间隔，有一些小时的数据是全部缺失的，比如7点的后一组数据是9点，8点的数据缺失了，此时两组
    的数据时间间隔为2 hours
    '''
    filearray = ['data/北京空气质量汇总.csv','data/北京气象数据汇总.csv']


    res=pd.read_csv(filearray[0])

    for i in range(1, len(filearray)):
        temp = pd.read_csv(filearray[i])
        res = pd.merge(res, temp,on='time') # 将文件拼接成一个

    res['time'] = pd.to_datetime(res['time'])
    res.set_index('time',inplace=True)
    train_dataset=res['2015':'2019']
    valid_dataset=res['2020':'2020']
    test_dataset=res['2021':'2021']

    x_train=[]
    y_train=[]
    x_valid=[]
    y_valid=[]
    x_test=[]
    y_test=[]

    X=[(x_train,y_train),(x_valid,y_valid),(x_test,y_test)]
    D=[train_dataset,valid_dataset,test_dataset]
    for j,(x,y) in enumerate(X):
        res=D[j]
        res.reset_index(inplace=True)
        for i in range(0, res.shape[0] - timestep - predict_step + 1):
            if check_data(res.loc[i:i + timestep + predict_step - 1], timestep, max_error, predict_step):
                x.append(np.array(res.loc[i:i + timestep - 1].drop('time', axis=1)))
                y.append(np.array(res.loc[i + timestep + predict_step - 1, predict_type]))
        x = np.array(x)
        y = np.array(y)



    return (x_train,y_train,x_valid,y_valid,x_test,y_test)

if __name__=='__main__':
    steps=[1,2,4,6,8,12,18,24]

    X_train={}
    Y_train={}
    X_valid={}
    Y_valid={}
    X_test={}
    Y_test={}

    for step in steps:
        x_train,y_train,x_valid,y_valid,x_test,y_test = get_data(timestep=24, predict_step=step,  max_error=pd.Timedelta('1 hours'))
        X_train[f'predict_{step}hours'] = x_train
        Y_train[f'predict_{step}hours'] = y_train
        X_valid[f'predict_{step}hours'] = x_valid
        Y_valid[f'predict_{step}hours'] = y_valid
        X_test[f'predict_{step}hours'] = x_test
        Y_test[f'predict_{step}hours'] = y_test
    np.savez('data/x_train.npz',X_train)
    np.savez('data/y_train.npz',Y_train)
    np.savez('data/x_valid.npz',X_valid)
    np.savez('data/y_valid.npz',Y_valid)
    np.savez('data/x_test.npz',X_test)
    np.savez('data/y_test.npz',Y_test)

