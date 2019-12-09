import os
import sys
import argparse
import numpy as np
from numpy import random
import function

def parse_args():
    parser = argparse.ArgumentParser(description="DPFTC")
    parser.add_argument('--loadfile', type=str, default='D:/DOCYJ/DPFTC/dataset2.csv',
                        help='Load file path.')
    parser.add_argument('--rank', type=int, default=1,
                        help='Tensor rank.')
    parser.add_argument('--feature', type=int, default=200,
                        help='Feature num.')
    parser.add_argument('--usernum', type=int, default=200,
                        help='User num.')
    parser.add_argument('--filetype', type=str, default='csv',
                        help='File Type.')
    parser.add_argument('--usercode', type=str, default='D:/DOCYJ/DPFTC/DESYNPUF_ID_Encode.csv',
                        help='User code File.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    loadfile = args.loadfile
    rank = args.rank
    feature_num = args.feature
    user_num = args.usernum
    filetype = args.filetype
    user_code_file = args.usercode
    if filetype == 'txt':
        LoadData,NodeNum,T_Num = function.TxtFileLoad(loadfile)
        print(NodeNum,T_Num)
        DataGraph = function.create_graph_by_time(LoadData,NodeNum,T_Num)
        DataTensor = function.create_tensor(random.rand(T_Num,NodeNum,rank),random.rand(T_Num,rank,feature_num))
        print(DataTensor)
    elif filetype == 'csv':
        LoadData,T_Num,min_date_str = function.CsvFileLoad(loadfile,user_num,user_code_file)
        U_Num, F_Num = np.shape(LoadData)
        DataGraph = function.create_graph2_by_time(LoadData, user_num, T_Num,min_date_str)
        DataTensor = function.create_tensor2(LoadData,np.zeros((T_Num, user_num, rank), dtype=complex),
                                             np.zeros((T_Num, rank, feature_num), dtype=complex),min_date_str)
        print(user_num,feature_num,T_Num,min_date_str)
        tensor_num = T_Num*user_num*feature_num
        number_num = 0
        for i in range(T_Num):
            for j in range(user_num):
                for k in range(feature_num):
                    if DataTensor[i,j,k] > 0:
                        number_num += 1
        print(number_num,tensor_num,round((number_num/tensor_num),4))
        # print(DataGraph)
        # print(DataTensor)



