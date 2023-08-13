import pandas as pd
import glob
from chinese_calendar import is_workday

#获取某一年空气质量数据的函数
def get_one_year_date(road):
    filearray = []
    filelocation = glob.glob(road+r'/*.csv')
    for filename in filelocation:
        filearray.append(filename)
    # 读取当前文件夹下的所有csv文件名称

    res = pd.read_csv(filearray[0])
    for i in range(1, len(filearray)):
        temp = pd.read_csv(filearray[i])
        res = pd.concat([res, temp], ignore_index=True, sort=False)
    # 将文件拼接成一个

    res = res[['type', 'date', 'hour', '北京']]
    res.rename({'北京': 'value'}, axis=1, inplace=True)#将“北京”索引改成“value”
    res['date'] = res['date'].astype('str')
    res['hour'] = res['hour'].astype('str')
    # 转成str格式

    res['date'] = pd.to_datetime(res['date'])
    res['year'] = res['date'].dt.year
    res['month'] = res['date'].dt.month
    res['day'] = res['date'].dt.day
    res['time'] = pd.to_datetime(pd.DataFrame([res['year'], res['month'], res['day'], res['hour']]).T)
    #将date和hour数据合并成time数据

    res = res.loc[:, ['type', 'time', 'value']]
    #获取此三栏数据

    return res

if __name__=='__main__':

    begin_year = 2015  # 开始的年份
    end_year = 2021  # 结束的年份

    dataset=pd.DataFrame()
    for i in range(begin_year,end_year+1):
        road=f'data/城市空气质量/城市_{i}0101-{i}1231'
        res=get_one_year_date(road)
        dataset = pd.concat([res, dataset], ignore_index=True, sort=False)



    dataset.set_index(['time', 'type'], inplace=True)
    # 设置time和type为行索引
    dataset=dataset.unstack()
    # 将污染物指标转成列索引
    dataset.columns=dataset.columns.droplevel(0)#去掉最外层的列索引
    dataset=dataset[['PM2.5','PM2.5_24h','PM10','PM10_24h','CO','CO_24h','NO2','NO2_24h','O3','O3_24h',
                     'SO2','SO2_24h']]
    dataset=dataset.dropna(how='all') # 删除全部的空行

    dataset['时间']=dataset.index
    dataset['年'], dataset['季节'],  dataset['小时'] = dataset['时间'].dt.year, dataset[
        '时间'].dt.quarter, dataset['时间'].dt.hour
    work_day=[]
    for day in dataset['时间']:
        if is_workday(day):
            work_day.append(1)
        else:
            work_day.append(0)
    dataset['工作日']=work_day
    dataset.drop('时间',axis=1,inplace=True)

    print('======数据缺失情况======')
    dataset.info()

    dataset.interpolate(axis=0,inplace=True) #用线性插值填充空缺值
    dataset=dataset.resample('1H').pad()
    dataset.interpolate(axis=0,inplace=True) #用线性插值填充空缺值
    dataset.to_csv('./data/北京空气质量汇总3.csv')
