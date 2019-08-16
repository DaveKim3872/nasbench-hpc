from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from nasbench import api
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import os

NASBENCH_TFRECORD = '../../nasbench-master/dataset/nasbench_only108.tfrecord'
dest = './dataset/destfile'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


def get_all_data(data_dir):
    # Load whole nasbench dataset
    nasbench = api.NASBench(data_dir)
    # Load the mapping from graph to accuracy
    dataMap = nasbench.get_maping()
    return dataMap


def matrix_eq_comp(mat1, mat2):
    inters = np.intersect1d(mat1, mat2)
    bc1 = np.bincount(mat1)
    bc2 = np.bincount(mat2)
    same_count_list = [min(bc1[x], bc2[x]) for x in inters]
    same_count = sum(same_count_list)
    return same_count


dataMapping = get_all_data(NASBENCH_TFRECORD)
adj0 = np.array(np.array(dataMapping[0]['module_adjacency']).reshape(1, -1)[0])
adj1 = np.array(np.array(dataMapping[1]['module_adjacency']).reshape(1, -1)[0])
# op0 = np.array(dataMapping[0]['module_operations'])
# op1 = np.array(dataMapping[1]['module_operations'])
# inters = np.intersect1d(adj0, adj1)
# bc1 = np.bincount(adj0)
# bc2 = np.bincount(adj1)
# same_count_list = [min(bc1[x], bc2[x]) for x in inters]
# same_count = sum(same_count_list)
# count = sum(adj0 != adj1) + sum(op0 != op1)
# print(same_count_list)
# print(same_count)
print(np.sum(adj0 == adj1))
# print(type(adj0))
# print(type(adj1))
