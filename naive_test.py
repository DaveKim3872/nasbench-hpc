import numpy as np
import json


def eq_elements(item0, item1):
    cnt = 0
    adj0 = np.array(item0['module_adjacency'])
    adj1 = np.array(item1['module_adjacency'])
    ope0 = np.array(item0['module_operations'])
    ope1 = np.array(item1['module_operations'])
    for j in range(0, 7):
        eq = sum(np.array(adj0[j]) != np.array(adj1[j]))
        cnt = cnt + eq
    eq = sum(np.array(ope0) != np.array(ope1))
    cnt = cnt + eq
    return cnt


op1 = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
op2 = ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']

mat0 = [[0, 1, 1, 1, 0, 1, 0],  # input layer
        [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
        [0, 0, 0, 1, 0, 0, 1],  # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
        [0, 1, 1, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
        [0, 0, 1, 0, 0, 0, 1],  # 3x3 max-pool
        [0, 0, 0, 0, 0, 0, 0]]  # output layer
it0 = {'module_adjacency': mat0, 'module_operations': op1}

mat1 = [[0, 1, 1, 0, 0, 1, 0],  # input layer
        [0, 0, 0, 1, 0, 0, 1],  # 1x1 conv
        [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
        [0, 0, 0, 0, 0, 0, 0]]  # output layer
it1 = {'module_adjacency': mat1, 'module_operations': op2}

# op1 = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
# op2 = ['input', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
# op1 = np.array(op1)
# op2 = np.array(op2)
#
# # mat0 = np.array(mat0)
# # mat1 = np.array(mat1)
# count = 0
# for i in range(0, 7):
#     # eq = sum(np.array(mat0[i]) != np.array(mat1[i]))
#     count = count + sum(np.array(mat0[i]) != np.array(mat1[i]))
# print(count)
set0 = [it0, it1]

mat2 = [[0, 1, 1, 1, 0, 1, 0],  # input layer
        [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
        [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
        [0, 0, 0, 0, 0, 0, 0]]  # output layer
op3 = ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
tem0 = {'module_adjacency': mat2, 'module_operations': op3}

mat2 = np.array(np.array(mat2).reshape(1, -1)[0])
mat0 = np.array(np.array(mat0).reshape(1, -1)[0])
inters = np.intersect1d(mat0, mat2)
bc1 = np.bincount(mat0)
bc2 = np.bincount(mat2)
same_count_list = [min(bc1[x], bc2[x]) for x in inters]
same_count = sum(same_count_list)
print(len(mat2) - same_count)
print(mat2 - mat0)
print(type(mat2))
print(type(mat0))
