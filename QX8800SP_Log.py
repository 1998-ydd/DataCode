import numpy as np
import pandas as pd
import scipy.linalg as sl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
plt.style.use('seaborn')
plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Marks:
    pass

class Angles:
    pass

class Offset:
    pass

class Diedata:
    pass

class YDdata:
    pass

class ProcessData:
    pass

def gendict(namelist:list)->dict:
    return {item:[] for item in namelist}


class SPLog:
    """
    LogParser 类用于解析日志文件，提取其中的关键信息。
    """
    def __init__(self, log_file):
        """
        初始化方法，接收一个日志文件路径作为参数。
        """
        self.log_file = log_file

    def extract_logs(self):
        """
        从日志文件中提取关键信息。
        返回一个包含提取信息的列表。
        """
        pass

def getdata_2time(df,time1,time2):
    return (df[(df.index > time1)
              & (df.index < time2)]).copy()

def calc_dist(df:pd.DataFrame,num:int,name:str):
    """
    根据输入的Mark点DataFrame和Mark点个数,
    返回计算阵列Mark的Mark间距.

    Parameters:
    df: 需要进行Mark圆间距计算的Mark坐标DataFrame.
    num: 阵列Mark点个数.
    name: 一般为UL:左上视, UR: 右上视, DL: 左下视, DR: 右下视

    Returns:
    计算列向圆间距与横向圆间距拼接在Mark坐标右侧.
    """
    if num == 16:
        for i in range(15):
            if (i + 1) %4 != 0:
                df[name+f'{i+1}-{i+2}Dist'] = ((df.iloc[:,i*2  ]-df.iloc[:,i*2+2])**2 + 
                                    (df.iloc[:,i*2+1]-df.iloc[:,i*2+3])**2)**0.5 
        #横向圆心距
        for i in range(12):
            df[name+f'{i+1}-{i+5}Dist'] = ((df.iloc[:,i*2  ]-df.iloc[:,i*2+8])**2 + 
                                      (df.iloc[:,i*2+1]-df.iloc[:,i*2+9])**2)**0.5       
        return df
    elif num == 25:
        for i in range(24):
            if (i + 1) %5 != 0:
                df[name+f'{i+1}-{i+2}Dist'] = ((df.iloc[:,i*2  ]-df.iloc[:,i*2+2])**2 + 
                                    (df.iloc[:,i*2+1]-df.iloc[:,i*2+3])**2)**0.5 
        #横向圆心距
        for i in range(20):
            df[name+f'{i+1}-{i+6}Dist'] = ((df.iloc[:,i*2  ]-df.iloc[:,i*2+10])**2 + 
                                      (df.iloc[:,i*2+1]-df.iloc[:,i*2+11])**2)**0.5
        return df

def describe_3s(df):
    des_df = df.describe()
    des_df.loc["range"] = des_df.loc['max']-des_df.loc['min']
    des_df.loc["3sigma"] = des_df.loc['std']*3
    return des_df

def calc_vib(df):
    num = np.arange(1, df.shape[0]+1)
    x = df.iloc[:,0::2]
    y = df.iloc[:,1::2]
    x_3s = (x.std()*3).to_numpy()
    y_3s = (y.std()*3).to_numpy()
    l_3s = (x_3s**2+y_3s**2)**0.5

    # 使用numpy的polyfit函数进行线性拟合，得到斜率和截距
    slope = np.polyfit(num, x, 2)
    slope2 = np.polyfit(num,y, 2)
    num = np.tile(num,(len(x_3s),1))
    # 计算拟合后的y值
    x_fit = slope[0,:] * num.T**2 + slope[1,:]*num.T
    y_fit = slope2[0,:] * num.T**2 + slope2[1,:]*num.T

    # 减去斜率进行平滑处理
    x_res = x - x_fit
    y_res = y - y_fit
    x_res_3s = (x_res.std()*3).to_numpy()
    y_res_3s = (y_res.std()*3).to_numpy()
    l_res_3s = (x_res_3s**2+y_res_3s**2)**0.5
    dataframe = pd.DataFrame({f'Marks':[f'第 {i} 个Mark' for i in range(1, len(x_3s)+1)],
                                  'X-3sigma':x_3s,
                                  'Y-3sigma':y_3s,
                                  'L-3sigma':l_3s,
                                  'X去温漂3s':x_res_3s,
                                  'Y去温漂3s':y_res_3s,
                                  'L去温漂3s':l_res_3s})
    return dataframe

def catdata_near(df1:pd.DataFrame,*dfs:pd.DataFrame)->pd.DataFrame:
    """
    以 df1 的时间为节点,拼接DataFrame

    Parameters:
    df1 (pd.DataFrame): 时间刻度DataFrame.
    dfs (pd.DataFrame): 需要拼接的DataFrames.

    Returns:
    将最接近 df1 的 dfs 数据拼接到 df1 上.
    """

    result = df1.copy() 
    try:
        for df in dfs:
            index = []
            for i in result.index:
                index.append(df[df.index < i].index[-1])
            df2 = df.loc[index]
            result = pd.concat([result, df2.set_index(result.index)], axis=1)
            #result[list(df2.columns)] = df2.values
        return result
    except:
        print(result.index[0],[df.index[0] for df in dfs])






# def get_mark(Dict,line,Dict1=None,Dict2=None,pattern=r'\d{3,}\.\d{6,}'):    
#     pattern = r'\d{3,}\.\d{6,}'
#     matches = re.findall(pattern, line)
#     keys = [i for i in Dict]
#     if len(matches) == 2 or len(matches) == 32:
#         Dict["time"].append(line.split()[0] + " " + line.split()[1])
#         for i,j in zip(keys[1:],matches):
#             Dict[i].append(j)

#     elif len(matches) == 40:
#         if Dict1:
#             keys40 = [i for i in Dict1]
#             Dict1["time"].append(line.split()[0] + " " + line.split()[1])
#             for i,j in zip(keys40[1:],matches):
#                 Dict1[i].append(j)

#     elif len(matches) == 50:
#         if Dict2:
#             keys50 = [i for i in Dict2]
#             Dict2["time"].append(line.split()[0] + " " + line.split()[1])
#             for i,j in zip(keys50[1:],matches):
#                 Dict2[i].append(j)

#     else:
#         print(line)
#         print(f"列表长度出错：{len(matches)}")

# def get_angle(Dict,line):
#     try:
#         pattern = r'[-]?\d{1,}.\d{6,}'
#         matches = re.findall(pattern, line)
#         keys = [i for i in Dict]
#         Dict[keys[1]].append(matches[0])
#         Dict["time"].append(line.split()[0] + " " + line.split()[1])
#     except Exception as e:
#         # 当发生异常时，执行此代码块
#         print(line)
#         print("发生了一个错误：", e)


def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False
    
