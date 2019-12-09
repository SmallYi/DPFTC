import os
import math
import csv
import datetime
from dateutil import rrule
import numpy as np
import pandas as pd
from numpy import random
from scipy.fftpack import fft, ifft
from federated import t_product


# 读取txt文件中的数据，处理成三元组列表形式[src,des,sec]
def TxtFileLoad(filepath):
    DataLoad = []
    max_id = -1
    max_T = -1
    file = open(filepath, "r")
    lines = file.readlines()
    for line in lines:
        datalist = list(map(int, line.split(' ')))
        large_id = max(datalist[0], datalist[1])
        now_T = math.ceil(datalist[2] / (3600 * 24))  # 向上取整
        if large_id > max_id:
            max_id = large_id
        if now_T > max_T:
            max_T = now_T
        DataLoad.append(datalist)
    file.close()
    return DataLoad, max_id + 1, max_T


# 读取csv文件中的数据，首先编码，再提取三元组和特征
def CsvFileLoad(filepath,user_num,user_code_file):
    data = pd.read_csv(filepath, low_memory=False)
    basic_info = np.array(data[['DESYNPUF_ID']])

    user_info = StrEncodeNum(basic_info[:,0], user_code_file,user_num)
    user_list = []
    for user in user_info:
        user_list.append(user[0])
    data_cut = DataframeCut(user_list,data)

    basic_info_cut = np.array(data_cut[['DESYNPUF_ID', 'CLM_FROM_DT',
                                        'CLM_PMT_AMT', 'AT_PHYSN_NPI', 'OP_PHYSN_NPI',
                                        'OT_PHYSN_NPI', 'ICD9_DGNS_CD_1', 'ICD9_DGNS_CD_2',
                                        'ICD9_DGNS_CD_3', 'ICD9_DGNS_CD_4', 'ICD9_DGNS_CD_5']])


    basic_info_cut[:,0] = StrEncodeCache(basic_info_cut[:,0]) # 对已选取用户临时编码
    max_T,basic_info_cut[:,1],min_date_str = GetT_ByMonth(basic_info_cut[:,1]) # 计算最大相差月数
    U_Num, F_Num = np.shape(basic_info_cut)
    basic_info_cut[:,2:F_Num] = Normalize(basic_info_cut[:,2:F_Num]) # 归一化，适用于数值型
    return basic_info_cut,max_T,min_date_str

# 根据相差月数获取时间间隔
def GetT_ByMonth(data):
    data_no_nan = []
    for i in range(len(data)):
        if math.isnan(data[i]):
            data[i] = 'nan'
        else:
            data_no_nan.append(int(data[i]))
            data[i] = str(int(data[i]))
    mx = str(max(data_no_nan))
    mn = str(min(data_no_nan))
    max_T = BtwMonth(mn,mx)
    return max_T,data,mn


def BtwMonth(start_str,end_str):
    v_year_end = int(end_str[0:4])
    v_year_start = int(start_str[0:4])
    v_month_end = int(end_str[4:6])
    v_month_start = int(start_str[4:6])
    v_day_end = int(end_str[6:8])
    v_day_start = int(start_str[6:8])
    interval = (v_year_end - v_year_start) * 12 + \
               (v_month_end - v_month_start) + \
               (v_day_end - v_day_start) * (1/30)
    return math.ceil(interval)




# 根据用户名选取对应行的数据
def DataframeCut(username_list,df):
    df_cut = df.loc[df['DESYNPUF_ID'].isin(username_list)]
    return df_cut

# 保存编码文件以便解码
def CsvEncodeSave(DataDict, filepath):
    datalist = []
    for value in DataDict.values():
        datalist.append(value)
    head = ['Name', 'EncodeID', 'OccurTimes']
    savadata = pd.DataFrame(columns=head, data=datalist)
    savadata.to_csv(filepath, encoding='gbk')
    encode_dict_sort = Array2Dict_Sort(datalist)
    return encode_dict_sort


# 对矩阵每一列单独进行归一化 np.array
def Normalize(matrix):
    U_Num, F_Num = np.shape(matrix)
    for i in range(F_Num):
        matrix[:, i],nan_cache,data_info = NormalPreDeal(matrix[:, i])
        m = np.mean(data_info)
        mx = max(data_info)
        mn = min(data_info)
        for j in range(U_Num):
            if j not in nan_cache:
                matrix[j, i] = ((float(matrix[j, i]) - mn) / (mx - mn))*10
    return matrix

# 归一化的预处理,某一列,如果为字符串就不处理，如果有nan先设置为0
def NormalPreDeal(data):
    nan_cache = [] # 保存nan的位置
    data_info = [] # 保存数值
    for i in range(len(data)):
        if type(data[i]) == type('str'):
            try:
                data[i] = float(data[i])
                data_info.append(data[i])
            except:
                if data[i].find('OTHER') != -1:
                    nan_cache.append(i)
                else:
                    data[i] = float(data[i][1:])
                    data_info.append(data[i])
        elif math.isnan(data[i]):
            nan_cache.append(i)
            data[i] = 'nan'
        else:
            data_info.append(data[i])
    return data,nan_cache,data_info


# 已获取出现次数较多的用户，对这些用户重新临时编码
def StrEncodeCache(datalist):
    data_cache = []
    i = 0
    for data in datalist:
        if data not in data_cache:
            data_cache.append(data)
        datalist[i] = data_cache.index(data)
        i += 1
    return datalist


# 利用列表索引对字符串编码,对整个源文件编码，记录用户出现次数并保存
def StrEncodeNum(datalist,filename,usernum):
    if os.path.exists(filename):
        print(filename,'exist')
        data = pd.read_csv(filename, low_memory=False)
        encode_info = np.array(data[['Name', 'EncodeID', 'OccurTimes']])
        encode_dict_sort = Array2Dict_Sort(encode_info)
    else:
        data_cache = []
        encode_record = {}
        i = 0
        for data in datalist:
            if data not in data_cache:
                data_cache.append(data)
                encode_record[data] = [data, data_cache.index(data), 1]
            encode_record[data][2] += 1
            datalist[i] = data_cache.index(data)
            print(data, 'encode to:', datalist[i])
            i += 1
        encode_dict_sort = CsvEncodeSave(encode_record, filename)
    if usernum <= len(encode_dict_sort):
        encode_dict_sort_cut = encode_dict_sort[0:usernum]
    else:
        encode_dict_sort_cut = encode_dict_sort
    return encode_dict_sort_cut


def DictCut(data,num):
    i = 0
    res = {}
    for key in data.keys():
        if i < num:
            res[key] = data[key]
        else:
            break
    return res

def Array2Dict_Sort(data):
    encode_dict = {}
    for item in data:
        encode_dict[item[0]] = [item[1],item[2]]
    sort_dict = sorted(encode_dict.items(),key=lambda code:code[1][1],reverse=True)
    return sort_dict


# 创建低秩张量
def create_tensor(A, B):
    [a3, a1, a2] = A.shape
    [b3, b1, b2] = B.shape
    A = fft(A, axis=0)
    B = fft(B, axis=0)
    C = np.zeros((b3, a1, b2), dtype=complex)
    for i in range(b3):
        C[i, :, :] = np.dot(A[i, :, :], B[i, :, :])
    C = ifft(C, axis=0)
    return C

# 创建csv低秩张量，根据特征填充张量,有日期才填充数据
def create_tensor2(LoadData,A,B,min_date_str):
    C = create_tensor(A,B)
    for data in LoadData:
        user = data[0]
        date_str = data[1]
        if date_str != 'nan':
            now_T = BtwMonth(min_date_str, date_str)
            for index in range(2,len(data)):
                if data[index] == 'nan' or data[index] == 'OTHER':
                    C[now_T-1,user,index] = 0
                else:
                    C[now_T-1,user,index] = data[index]
    return C


def product_tensor(A, B):
    return t_product(A, B)


# 创建时段图
def create_graph_by_time(LoadData, NodeNum, T_Num):
    graph = np.zeros((T_Num, NodeNum, NodeNum), dtype=complex)
    for data in LoadData:
        src, des, sec = data
        now_T = math.ceil(sec / (3600 * 24))  # 向上取整
        graph[now_T - 1, src, des] = 1
    return graph

# 创建csv文件的时段图
def create_graph2_by_time(LoadData, NodeNum, T_Num,min_date_str):
    graph = np.zeros((T_Num, NodeNum, NodeNum), dtype=complex)
    for data in LoadData:
        user = data[0]
        date_str = data[1]
        if date_str != 'nan':
            now_T = BtwMonth(min_date_str,date_str)
            graph[now_T - 1, user, user] = 1
    return graph


# 创建邻接矩阵
def create_adjacent(nNode, k, densityRate, unique):
    adjacent = np.zeros((k, nNode, nNode), dtype=complex)
    if unique:
        for i in range(k):
            if i == 0:
                for iNode in range(math.floor((nNode + 1) / 2)):
                    for jNode in range(iNode + 1, nNode):
                        if random.rand(1) < densityRate:
                            adjacent[i, iNode, jNode] = 1
                            adjacent[i, jNode, iNode] = 1
            else:
                adjacent[i, :, :] = adjacent[0, :, :]
    else:
        for i in range(k):
            for iNode in range(int(math.floor((nNode + 1) / 2))):
                for jNode in range(iNode + 1, nNode):
                    if random.rand(1) < densityRate:
                        adjacent[i, iNode, jNode] = 1
                        adjacent[i, jNode, iNode] = 1
    return adjacent
