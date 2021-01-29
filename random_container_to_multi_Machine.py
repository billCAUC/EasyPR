#!/usr/bin/env python
# coding: utf-8

# In[78]:


'''
Author: 娄炯
Date: 2021-01-28 12:49:04
LastEditors: loujiong
LastEditTime: 2021-01-28 18:47:36
Description: 
Email:  413012592@qq.com
'''
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import pickle
import pathlib
import networkx as nx
import collections
import queue
import copy

random.seed(0)


class Trace:
    def __init__(self):
        self.trace_file = "trace.pkl"
        self.task_list = []
        self.allContainer = []
        self.allLayer = []
        self.image_stats = dict()
        self.image_name_list = []
        '''
        layer_stats key 为layer名字（hash码） value为list
        value的list当中包含[star数量,下载次数,被多少个image（只统计了我们下载的image）包含,大小, 编号,具体参与到的image名字]
        'sha256:8d380c957e3e85d308c5520a5fa0a29687cf3bb956cd141b5fc14c78cf9dfed1': ['110', '159234', '1', '639052673',10,['swift']]
        '''
        self.layer_stats = dict()

        with open("pull/image_stats.csv", "r") as f:
            cnt = 0
            for lines in f.readlines():
                image_item = lines[:-1].split(",")
                self.image_stats[image_item[0]] = image_item[1:]
                self.image_stats[image_item[0]].append(cnt)
                self.allContainer.append(self.image_stats[image_item[0]])
                self.image_name_list.append(image_item[0])
                cnt += 1
                #print(self.image_stats[image_item[0]])
        with open("pull/layer_stats.csv", "r") as f:
            cnt = 0
            for lines in f.readlines():
                layer_item = lines[:-1].split(",")
                self.layer_stats[layer_item[0]] = layer_item[1:-1]
                self.layer_stats[layer_item[0]].append(cnt)
                self.layer_stats[layer_item[0]].append(
                    [i[8:] for i in layer_item[-1].split("|")])
                for i in layer_item[-1].split("|"):
                    if len(self.image_stats[i[8:]]) == 5:
                        self.image_stats[i[8:]].append([])
                    self.image_stats[i[8:]][5].append(layer_item[0])
                self.allLayer.append(self.layer_stats[layer_item[0]])
                #print(self.layer_stats[layer_item[0]])
                cnt += 1
        '''
        image_name_list 包含所有image名字的list
        ['node-env', 'ubuntu-upstart', 'iojs', 'ruby-env', 'perl',...]
        '''


def O2ST(original_G, lambd, infinity=10000000):
    ST_G = nx.DiGraph()
    ST_G.add_node('s')
    ST_G.add_node('t')
    for i in original_G.nodes():
        ST_G.add_edge('s', i, capacity=lambd * original_G.nodes()[i]['w'])
        ST_G.add_edge(i, 't', capacity=original_G.nodes()[i]['p'])
    for i in original_G.nodes():
        for j in original_G.nodes():
            if original_G.has_edge(i, j):
                ST_G.add_edge(j, i, capacity=infinity)
    return ST_G


def calGRank(graph):
    sum_p = 0
    sum_w = 0
    for i in graph.nodes():
        sum_p = sum_p + graph.nodes()[i]['p']
        sum_w = sum_w + graph.nodes()[i]['w']
        # print(i)
    res = sum_p / sum_w
    return res

def calMinRank(graph, infinity= 1000000):
    rank = infinity
    for i in graph.nodes():
        if ( graph.nodes()[i]['w'] > 0 and rank > graph.nodes()[i]['p'] / graph.nodes()[i]['w']):
            rank = graph.nodes()[i]['p'] / graph.nodes()[i]['w']
    return rank


def calRank(graph, partition):
    # partition为set
    if ('s' in partition):
        partition.remove('s')
        sum_p = 0
        sum_w = 0
        l_partition = list(partition)
        for i in l_partition:
            sum_p = sum_p + graph.node[l_partition[i]]['p']
            sum_w = sum_w + graph.node[l_partition[i]]['w']
        res = sum_p / sum_w
    else:
        return -1
    return res

class Machine:
    def __init__(self):
        self.layer_list = []
        self.container_list = []
        
# sort group
def cal_group_size(subgroup,tr):
    allLayer = set()
    for container in subgroup:
        allLayer = allLayer|{_layer_name for _layer_name in tr.image_stats[container][5]}
    sumSize = sum([int(tr.layer_stats[_layer_name][3]) for _layer_name in allLayer])
    return sumSize
def sortGroup(group_list1,tr):
    group_list = copy.deepcopy(group_list1)
    n = len(group_list)
    for i in range(n):
        for j in range(0, n-i-1):
            if cal_group_size(group_list[j],tr) > cal_group_size(group_list[j+1],tr):
                group_list[j], group_list[j+1] = group_list[j+1], group_list[j]
    return group_list
def find_min(matrix_machine_container_time):
    min_Score = []
    for i in range(len(matrix_machine_container_time)):
        min_Score.append(min(zip(matrix_machine_container_time[i].values(),matrix_machine_container_time[i].keys())))
    min_value = 100000000000000000000000
    min_machine_index = 0
    min_container_name = 0
    for i in range(len(min_Score)):
        if(min_Score[i][0]<min_value):
            min_value = min_Score[i][0]
            min_index = i
            min_container_name =min_Score[i][1]
    return min_value,min_index,min_container_name


# In[81]:


machine_number = 20
    
tr = Trace()
container_number = random.randint(50, 100)
container_list = [
    random.sample(tr.image_name_list, 1)[0]
    for i in range(container_number)
]
container_list_unique = list()
for i in container_list:
    if i not in container_list_unique:
        container_list_unique.append(i)
container_counter = collections.Counter()
container_counter.update(container_list)
container_size = {i:0 for i in container_counter}

comprised_layer_list = list()
for _container_name in container_counter:
    for _layer_name in tr.image_stats[_container_name][5]:
        if _layer_name not in comprised_layer_list:
            comprised_layer_list.append(_layer_name)
        container_size[_container_name] +=  int(tr.layer_stats[_layer_name][3])
original_G = nx.DiGraph()
    # original graph
    
original_G.add_nodes_from([(_container_name, {
    "p": 0,
    "w": container_counter[_container_name]
}) for _container_name in container_counter])


original_G.add_nodes_from([(_layer_name, {
    "p": int(tr.layer_stats[_layer_name][3]) / 10000000,
    "w": 0
}) for _layer_name in comprised_layer_list])

edges_list = []
for _container_name in container_counter:
    for _layer_name in tr.image_stats[_container_name][5]:
        edges_list.append((_layer_name , _container_name))
original_G.add_edges_from(edges_list)

group_list = []
while (len(original_G.nodes()) != 0):
    lambda_min = calMinRank(original_G)
    lambda_max = calGRank(original_G)
    left = lambda_min
    right = lambda_max
    lS = 1
    while (lS == 1):
        mid = (left + right) / 2
        st_G = O2ST(original_G, mid)
        _, partition = nx.minimum_cut(st_G, "s", "t")
        if len(partition[0]) == 1:
            left = mid
        lS = len(partition[0])
        
    group_list.append(partition[0])
    original_G.remove_nodes_from(partition[0])
group_queue = queue.Queue()           # 所有group
for index,item in enumerate(group_list):
    _container_in_group = []
    for j in item:
        if j in container_counter:
            _container_in_group.append(j)
    group_list[index] = _container_in_group
    group_queue.put(_container_in_group)
    
machine_list = [Machine() for i in range(machine_number)]
download_size_per_machine = sum([int(tr.layer_stats[_layer_name][3]) for _layer_name in comprised_layer_list])/machine_number
    
scheduled_size = 0  
scheduled_flag = {i:0 for i in container_counter}
current_group = []                   # 当前group
scheduled_layer = []
C_time_list = dict()

while(len(group_list)!=0):     
# sort group
    group_list = sortGroup(group_list,tr)
    subgroup = group_list[0]
    while(len(subgroup) != 0):
        matrix_machine_container_time = []
        for _machine_index in range(machine_number):
            matrix_machine_container_time.append([])
            matrix_machine_container_time[_machine_index] = dict()
            for container_index in subgroup:
                matrix_machine_container_time[_machine_index][container_index] = sum([int(tr.layer_stats[_layer_name][3]) for _layer_name in tr.image_stats[container_index][5] if _layer_name not in machine_list[_machine_index].layer_list])   
        # find min_container_machine
        min_value,min_machine,min_container  = find_min(matrix_machine_container_time)
        # put container in machine    
        machine_list[_machine_index].container_list.append(min_container)
        for _layer_name in  tr.image_stats[min_container][5]:
            if _layer_name not in scheduled_layer:
                scheduled_layer.append(_layer_name)

        for _layer_name in  tr.image_stats[min_container][5]:
            if _layer_name not in machine_list[_machine_index].layer_list:
                machine_list[_machine_index].layer_list.append(_layer_name)
        # delete container 
        scheduled_flag[min_container] = 1 
        C_time_list[min_container] = container_counter[min_container] * sum([int(tr.layer_stats[_layer_name][3])for _layer_name in machine_list[_machine_index].layer_list])
        scheduled_size = sum([int(tr.layer_stats[_layer_name][3]) for _layer_name in scheduled_layer])
        subgroup.remove(min_container)
        if(len(group_list[0])==0):
            group_list.pop(0)

#print(C_time_list)
#print(sum(list(C_time_list.values())))
#a = sum(list(C_time_list.values()))
print([machine.container_list for machine in machine_list])
#print(a)


# In[ ]:




