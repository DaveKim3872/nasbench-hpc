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


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def print_sample(item):
    print("++++++++++++++SAMPLE++++++++++++++++")
    for i in range(0, 5):
        print(item[i])
    print("+++++++++++SAMPLE_END+++++++++++++++")


def is_included(item, item_list):
    for temp in item_list:
        # print(item)
        # print(temp)
        if len(np.array(item['module_adjacency'])) == len(np.array(temp['module_adjacency'])):
            if ((np.array(item['module_adjacency']) == np.array(temp['module_adjacency'])).all()) \
                    and \
                    ((np.array(item['module_operations']) == np.array(temp['module_operations'])).all()):
                return True
    return False


def remove_array(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def matrix_eq_comp(mat1, mat2):
    inters = np.intersect1d(mat1, mat2)
    bc1 = np.bincount(mat1)
    bc2 = np.bincount(mat2)
    same_count_list = [min(bc1[x], bc2[x]) for x in inters]
    same_count = sum(same_count_list)
    return same_count


def one_dif(item0, item1):
    adj0 = np.array(item0['module_adjacency']).reshape(1, -1)[0]
    adj1 = np.array(item1['module_adjacency']).reshape(1, -1)[0]
    op0 = np.array(item0['module_operations'])
    op1 = np.array(item1['module_operations'])
    length0 = len(op0)
    length1 = len(op1)
    if length0 != length1:
        return False
    count = sum(op0 != op1)
    if count > 1:
        return False
    else:
        count = count + (len(adj0) - matrix_eq_comp(adj0, adj1))
        if count != 1:
            return False
    return True


def get_all_data(data_dir):
    # Load whole nasbench dataset
    nasbench = api.NASBench(data_dir)
    # Load the mapping from graph to accuracy
    dataMapping = nasbench.get_maping()
    return dataMapping


def find_next_best(data_set, current_best, start):
    # remove_array(data_set, current_best)
    reducing_set = data_set
    reduced_set = []
    reduced_set = data_set
    for item in reducing_set[:]:
        if one_dif(item, current_best):
            # print(item)
            if item['final_test_accuracy'] > current_best['final_test_accuracy']:
                remove_array(reducing_set, current_best)
                reduced_set, current_best = find_next_best(reducing_set, item)
    return reduced_set, current_best


def main(argv):
    del argv

    dataMapping = get_all_data(NASBENCH_TFRECORD)
    data_index = 0
    for data in dataMapping:
        data['index'] = data_index
        data['next_state'] = -1
        data_index += 1
    print("overall_index is: ") + str(data_index)
    reduced_set = dataMapping
    current_best = dataMapping[0]
    cnt = 0
    for item in dataMapping:
        if is_included(item, reduced_set):
            print("for current adjacency:")
            print(item["module_adjacency"])
            print("and operations")
            print(item['module_operations'])
            print("whose accuracy is: " + str(item['final_test_accuracy']))
            reduced_set, current_best = find_next_best(reduced_set, item, true)
            # item['next_state'] = current_best['index']
            print("the next best is")
            print(current_best['module_adjacency'])
            print(current_best['module_operations'])
            print("whose accuracy is: " + str(current_best['final_test_accuracy']))
            print("current dataset length is: " + str(len(reduced_set)))
            cnt += 1
        else:
            print("No Match")
            cnt += 1
        if cnt % 100 == 0:
            dest_file = dest + "_length_" + str(len(reduced_set)) + "_cnt" + str(cnt) + ".json"
            output_file = open(dest_file, 'w+', encoding='utf-8')
            for dic in reduced_set:
                json.dump(dic, output_file, cls=MyEncoder)
                output_file.write("\n")
            output_file.close()


# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
    app.run(main)
