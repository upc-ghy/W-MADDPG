#!./venv python
# -*- encoding: utf-8 -*-
"""
@File    :   experiment_init.py    
@Contact :   neard.ws@gmail.com
@Github  :   neardws

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/8/31 下午4:35   neardws      1.0         None
"""
import pickle
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
# import h5py
from config.config import settings
from experiment.experiment_save_and_reload import save_experiment_median_to_pickle
from init_input.experiment_input_save_and_reload import load_pickle
from init_input.init_distance import get_task_id_under_edge_node_15, get_task_time_limitation_under_edge_node_15

def init_useful_channel_10(node_type, node_id, fixed_edge_node):
    if node_type == settings.NODE_TYPE_BASE_STATION:
        node = fixed_edge_node[node_id]
        node_channel = node["sub_channel"]
        channel_status = np.zeros(len(node_channel))
        useful_channel = {"node_channel": node_channel, "channel_status": channel_status}
        return useful_channel
    elif node_type == settings.NODE_TYPE_RSU:
        node = fixed_edge_node[node_id]
        node_channel = node["sub_channel"]
        channel_status = np.zeros(len(node_channel))
        useful_channel = {"node_channel": node_channel, "channel_status": channel_status}
        return useful_channel
    else:
        raise ValueError("from IPUSS.init_userful_channel 节点类型出错， 不是指定类型")

def get_usable_channel_list(useful_channel):
    """
    :argument
        useful_channel
    :return
        usable_channel_list     当前可用的信道
    """
    usable_channel_list = []

    channel_status = useful_channel["channel_status"]
    node_channel = useful_channel["node_channel"]
    for i in range(len(node_channel)):
        if channel_status[i] == 0:
            usable_channel_list.append(node_channel[i])

    return usable_channel_list

def get_combination_and_strategy_length_10(usable_channel_list_len, task_id_under_edge_node, time_limitation_under_edge_node):
    if len(task_id_under_edge_node) != 0:
        combination_of_task_and_time = [[-1, -1]]
        for i, task_id in enumerate(task_id_under_edge_node):
            combination_of_task_and_time.append([task_id, 2000])
        length_of_combination = int(len(combination_of_task_and_time)),
        print("length_of_combination")
        print(length_of_combination)
        print("usable_channel_list_len")
        print(usable_channel_list_len)
        print("np.power(length_of_combination, usable_channel_list_len)")
        print(np.power(np.int64(length_of_combination), np.int64(usable_channel_list_len)))
        return {"combination_of_task_and_time": combination_of_task_and_time,
                "length_of_combination": length_of_combination,
                "length_of_strategy_list": np.power(np.int64(length_of_combination), np.int64(usable_channel_list_len))}
    else:
        return

def print_to_console(msg, objective=None):
    print("*" * 32)
    print(msg)
    if objective is not None:
        print(objective)
        print(type(objective))


def save(iteration):
    # 读取实验参数设置
    # fixed_edge_node = None
    # edge_vehicle_node = None
    # task_by_time_list = None
    # fixed_distance_matrix_list = None
    # mobile_distance_matrix_list = None

    pickle_file = Path(load_pickle(1))
    # with pickle_file.open("rb") as fp:

    fp = pickle_file.open("rb")
    fixed_edge_node = pickle.load(fp)
    print_to_console("fixed_edge_node", fixed_edge_node)

    # edge_vehicle_node = pickle.load(fp)
    # print_to_console("edge_vehicle_node", edge_vehicle_node)

    task_by_time_list = pickle.load(fp)
    print_to_console("task_by_time_list", task_by_time_list)

    fixed_distance_matrix_list = pickle.load(fp)
    print_to_console("fixed_distance_matrix_list", fixed_distance_matrix_list)

    # mobile_distance_matrix_list = pickle.load(fp)
    # print_to_console("fixed_distance_matrix_list", fixed_distance_matrix_list)

    """
           ————————————————————————————————————————————————————————————————————————————————————
               实验进行初始化
               1、初始化迭代次数
               2、初始化边缘节点数量
               3、初始化激励值集合
               4、初始化可用信道
               5、初始化每个节点下的任务
               6、初始化节点当前可用信道列表
               7、初始化策略空间
               8、初始化选择概率
           ————————————————————————————————————————————————————————————————————————————————————
       """
    # 初始化迭代次数
    # task_on_base = []
    # for i in range(len(fixed_distance_matrix_list)):
    #     tmp_task_on_base = []
    #     task_on_base.append(tmp_task_on_base)
    # for i in range(len(fixed_distance_matrix_list)):
    #     for j in range(len(fixed_distance_matrix_list[i])):
    #         if fixed_distance_matrix_list[i][j] <= 500:
    #             task_on_base[i].append(j)
    # print_to_console("task_on_base", task_on_base)

    fixed_distance_matrix = fixed_distance_matrix_list
    # mobile_distance_matrix = mobile_distance_matrix_list[iteration]
    task_list = task_by_time_list
    # print_to_console("task_list[0]", task_list[0])
    # 初始化边缘节点数量
    node_num = settings.RSU_NUM_15
    # fixed_node_num = settings.BASE_STATION_NUM + settings.RSU_NUM
    # mobile_node_num = settings.EDGE_VEHICLE_NUM

    # 初始化激励值集合
    max_potential_value = np.zeros(len(fixed_edge_node))

    # 初始化可用信道
    useful_channel_under_node = []
    for i in range(settings.RSU_NUM_15):
        useful_channel_under_node.append(
            init_useful_channel_10(settings.NODE_TYPE_RSU, i, fixed_edge_node))

    # 初始化每个节点下的任务
    task_id_under_each_node_list = []

    for i in range(settings.RSU_NUM_15):
        print_to_console("初始化每个节点下的任务 RSU " + str(i))
        task_id_under_edge_node = get_task_id_under_edge_node_15(node_type=settings.NODE_TYPE_RSU,
                                                              node_id=i,
                                                              distance_matrix=fixed_distance_matrix)
        task_id_under_each_node_list.append(task_id_under_edge_node)


    # 显示节点下所有任务的覆盖情况
    # union_set = set()
    # for i, task_id_under_edge_node in enumerate(task_id_under_each_node_list):
    #     union_set = union_set | set(task_id_under_edge_node)
    # print("显示节点下所有任务的覆盖情况")
    # print(union_set)
    # print(len(union_set))

    # 初始化节点当前可用信道列表
    usable_channel_of_all_nodes = []

    for i in range(settings.RSU_NUM_15):
        print_to_console("初始化节点当前可用信道列表 RSU " + str(i))
        usable_channel = get_usable_channel_list(useful_channel_under_node[i])
        usable_channel_of_all_nodes.append(usable_channel)
    # print_to_console("fixed_distance_matrix_list[2][27]", fixed_distance_matrix_list[2][27])

    # 初始化任务的时间限制
    task_time_limitation_of_all_nodes = []
    for i in range(settings.RSU_NUM_15):
        # print_to_console("settings.RSU_NUM_10", settings.RSU_NUM_10)
        print_to_console("初始化任务的时间限制 RSU " + str(i))
        task_time_limitation_under_edge_node = get_task_time_limitation_under_edge_node_15(
            iteration=iteration,
            node_type=settings.NODE_TYPE_RSU,
            node_id=i,
            distance_matrix_list=fixed_distance_matrix_list,
            task_list=task_list)
        task_time_limitation_of_all_nodes.append(task_time_limitation_under_edge_node)
    # print_to_console("task_time_limitation_of_all_nodes",task_time_limitation_of_all_nodes)


    # 初始化所有节点的策略组合及策略列表长度
    combination_and_strategy_length_of_all_nodes = []
    print_to_console("usable_channel_of_all_nodes", usable_channel_of_all_nodes)
    for i in range(settings.RSU_NUM_15):
        print_to_console("初始化所有节点的策略组合及策略列表长度 RSU " + str(i))
        print(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
        print_to_console("i", i)
        print_to_console("usable_channel_of_all_nodes[i]",usable_channel_of_all_nodes[i])
        combination_and_strategy_length = get_combination_and_strategy_length_10(
            usable_channel_list_len=len(usable_channel_of_all_nodes[i]),
            task_id_under_edge_node=task_id_under_each_node_list[i],
            time_limitation_under_edge_node=task_time_limitation_of_all_nodes[i])
        combination_and_strategy_length_of_all_nodes.append(combination_and_strategy_length)
        print(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    print_to_console("fixed_node_num",node_num)

    save_success = save_experiment_median_to_pickle(iteration,
                                                    fixed_edge_node,
                                                    fixed_distance_matrix,
                                                    task_list,
                                                    node_num,
                                                    max_potential_value,
                                                    useful_channel_under_node,
                                                    task_id_under_each_node_list,
                                                    usable_channel_of_all_nodes,
                                                    task_time_limitation_of_all_nodes,
                                                    combination_and_strategy_length_of_all_nodes,
                                                    )
    if save_success:
        print("保存实验中间值成功")


if __name__ == '__main__':

    save(iteration=0)
    # save(iteration=1)
    # save(iteration=2)
    # save(iteration=3)
    # save(iteration=4)

