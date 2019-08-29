from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from nasbench import api
from collections import deque
import numpy as np
import json
import copy

NASBENCH_TFRECORD = 'D:/dataset/nasbench/nasbench_only108.tfrecord'
best_500_path = 'dataset/random500.txt'
best_500_res = 'dataset/rand500res.json'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OP_SPACE = [CONV1X1, CONV3X3, MAXPOOL3X3]


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def bypass_rules(index, adj):
    adj_len = len(adj)
    op_len = len(adj[0])
    if adj[0][index] == 1 and adj[index][op_len - 1] == 1:
        for j in range(1, adj_len - 1):
            if adj[j][index] == 1:
                return False
        return True
    return False


def output_rules(index, adj_larger):
    len_larger = len(adj_larger)
    new_node = np.zeros(len_larger)
    new_node[len_larger - 1] = 1
    if (adj_larger[index] == new_node).all():
        return True
    return False


def valid_size_transfer(larger, smaller):
    adj_larger = larger['module_adjacency'].tolist()
    adj_smaller = smaller['module_adjacency'].tolist()
    op_larger = larger['module_operations']
    op_smaller = smaller['module_operations']
    # bypass
    for i in range(1, len(op_larger)):
        if op_larger[i] in OP_SPACE:
            temp_list = copy.deepcopy(op_larger)
            temp_list.pop(i)
            if temp_list == op_smaller and bypass_rules(i, adj_larger):
                del_adj = copy.deepcopy(adj_larger)
                del_adj.pop(i)
                for temp_node in del_adj[:]:
                    del temp_node[i]
                if del_adj == adj_smaller:
                    return True
            # new output
            elif temp_list == op_smaller and output_rules(i, adj_larger):
                for j in range(0, len(adj_larger)):
                    if j != i and adj_larger[j][len(adj_larger) - 1] == 1:
                        return False
                del_adj = copy.deepcopy(adj_larger)
                del_adj.pop(i)  # size: bigger_size * (bigger_size - 1) [new node deleted]
                for k in range(0, len(adj_smaller)):
                    if del_adj[k][i] != adj_smaller[k][len(op_smaller) - 1]:
                        return False
                    del_adj[k][len(op_larger) - 1] = del_adj[k][i]
                    del del_adj[k][i]
                if del_adj == adj_smaller:
                    return True
    return False


def one_dif(item0, item1):
    adj0 = np.array(np.array(item0['module_adjacency']).reshape(1, -1)[0])
    adj1 = np.array(np.array(item1['module_adjacency']).reshape(1, -1)[0])
    op0 = np.array(item0['module_operations'])
    op1 = np.array(item1['module_operations'])
    length0 = len(op0)
    length1 = len(op1)
    # count = 0
    # if length0 - length1 == 1:  # add a node to transfer from item1 to item 0
    #     return valid_size_transfer(item0, item1)
    # elif length1 - length1 == 1:
    #     return valid_size_transfer(item1, item0)
    # elif length0 == length1:
    if length0 == length1:
        count = np.sum(op0 != op1)
        if count > 1:
            return False
        else:
            count = count + np.sum(adj0 != adj1)
            if count != 1:
                return False
        return True
    else:
        return False


def get_all_data(data_dir):
    # Load whole nasbench dataset
    nasbench = api.NASBench(data_dir)
    # Load the mapping from graph to accuracy
    dataMapping = nasbench.get_maping()
    return dataMapping


def get_neighbours(current_set, current_state, current_depth):
    count = 0
    neighbours = []
    for item in current_set[:]:
        if item['visited'] == 0:
            count += 1
            print("enter one_dif for " + str(count) + " times.")
            if one_dif(current_state, item):
                item['visited'] = 1
                item['depth'] = current_depth + 1
                # print("Item " + str(item['index']) + " depth is changed to " + str(item['depth']))
                neighbours.append(item)
    print("Got " + str(len(neighbours)) + " neighbours this time.")
    return neighbours, current_set


# using BFS to search
def find_trace_to_best(data_set, current_state, destination_index):
    print("Processing a BFS...")
    depth = 0
    count = 0
    search_queue = deque()
    neighbours, data_set = get_neighbours(data_set, current_state, depth)
    search_queue += neighbours
    searched = []
    while search_queue:
        count += 1
        checking_state = search_queue.popleft()
        depth = checking_state['depth']
        print("Searching in depth " + str(depth) + ", expanding " + str(count) + " nodes.")
        if checking_state['index'] not in searched:
            print("checking unsearched with index " + str(checking_state['index']))
            if checking_state['index'] == destination_index:
                print("Finish the BFS.")
                return depth
            else:
                print("Not match. Move to next.")
                neighbours, data_set = get_neighbours(data_set, checking_state, depth)
                search_queue += neighbours
                searched.append(checking_state)
    print("Error: Search failed.")
    return depth


def main(argv):
    del argv
    dataMapping = get_all_data(NASBENCH_TFRECORD)
    best_accuracy = 0
    best_accuracy_index = -1
    initial_states = []

    # initialize index
    for i in range(0, len(dataMapping)):
        dataMapping[i]['index'] = i
        dataMapping[i]['depth'] = 0
        dataMapping[i]['visited'] = 0

    # find the state with the best accuracy
    for state in dataMapping:
        # if state['final_test_accuracy'] > best_accuracy:
        if len(state['module_operations']) == 6 and state['final_test_accuracy'] > best_accuracy:
            best_accuracy = state['final_test_accuracy']
            best_accuracy_index = state['index']
    print("State " + str(best_accuracy_index) + " has the best accuracy of " + str(best_accuracy))

    # fetch 500 initial states
    with open(best_500_path, 'r') as f:
        best_500_hash = f.read().splitlines()
    for hash_key in best_500_hash:
        for data in dataMapping:
            if hash_key == data['hash_key']:
                full_data = copy.deepcopy(data)
                initial_states.append(full_data)
    print("Initialization completed.")

    # Width-First Search
    for state in initial_states:
        if len(state['module_operations']) != 6:
            continue
        searchMapping = copy.deepcopy(dataMapping)
        gt_search4best = {'initial_state': state['index'], 'initial_adj': state['module_adjacency'],
                          'initial_op': state['module_operations'],
                          'search_depth': find_trace_to_best(searchMapping, state, best_accuracy_index)}
        with open(best_500_res, 'w+', encoding='utf-8') as f:
            json.dump(gt_search4best, f, cls=MyEncoder)
            f.write("\n")
        return


# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
    app.run(main)
