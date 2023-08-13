import pandas as pd
import numpy as np

if __name__=='__main__':

    begin_year = 2015  # 开始的年份
    end_year = 2021  # 结束的年份

    dataset=pd.DataFrame()
    for i in range(begin_year,end_year+1):
        road=f'data/北京气象数据/{i}-01-01至{i}-12-31.xlsx'
        temp=pd.read_excel(road)
        temp = temp.iloc[:, [1, 3, 4, 5, 7, 9, 10, 12, 13]]

        dataset = pd.concat([dataset, temp], ignore_index=True, sort=False)


    dataset['time']=pd.to_datetime(dataset['时间'])
    temp = dataset[['time', '风向角度(度)']]
    temp.set_index('time', inplace=True)
    print('======风向角度数据缺失情况======')
    temp.info()
    temp.ffill(inplace=True)
    temp = temp.resample('1H').ffill()  # 按每个小时重采样，前向填充

    dataset.drop(labels=['时间','风向角度(度)'],axis=1,inplace=True)

    dataset.set_index('time',inplace=True)
    dataset['过去6小时降水量(mm)']=dataset['过去6小时降水量(mm)'].fillna(0)
    print('======数据缺失情况======')
    dataset.info()
    dataset.interpolate(axis=0,inplace=True) #用线性插值填充空缺值

    dataset=dataset.resample('1H').interpolate(axis=0)# 按每个小时重采样
    dataset=pd.merge(dataset, temp,on='time') # 将文件拼接成一个
    dataset.info()
    dataset.to_csv('./data/北京气象数据汇总.csv')

