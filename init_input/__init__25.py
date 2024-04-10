from init_input.experiment_input_save_and_reload import save_pickle
from init_input.init_distance import get_fixed_distance_matrix, get_mobile_distance_matrix
from init_input.init_edge_node import init_fixed_edge_node_25, init_edge_vehicle_node
from config.config import settings
from init_input.init_task_by_time import init_task_by_time_25
from init_input.init_vehicles import get_edge_vehicle_id, get_customer_vehicle_id, get_vehicle_id_25
from tqdm import tqdm
import multiprocessing
import pandas as pd
import numpy as np


def print_to_console(msg, objective=None):
    print("*" * 32)
    print(msg)
    if objective:
        print(object)
        print(type(object))


def process_wrapper(customer_vehicle_id, fixed_edge_node):
    # print_to_console("Processing task and distance matrix in time " + str(time))

    task_by_time = init_task_by_time_25(customer_vehicle_id=customer_vehicle_id)
    # print_to_console("task_by_time",task_by_time)

    fixed_distance_matrix = get_fixed_distance_matrix(fixed_edge_node=fixed_edge_node,
                                                      task_list=task_by_time)

    # mobile_distance_matrix = get_mobile_distance_matrix(time=time,
    #                                                     edge_vehicle_node=edge_vehicle_node,
    #                                                     task_list=task_by_time)
    print_to_console("task_by_time",task_by_time)
    return {"task_by_time": task_by_time, "fixed_distance_matrix": fixed_distance_matrix}


if __name__ == '__main__':
    # print_to_console("Start")

    fixed_edge_node = init_fixed_edge_node_25()

    # edge_vehicle_id = get_edge_vehicle_id()
    # edge_vehicle_node = init_edge_vehicle_node(edge_vehicle_num=settings.edge_vehicle_num,
    #                                            edge_vehicle_id=edge_vehicle_id)

    # print_to_console("Edge Node Inited")

    customer_vehicle_id = get_vehicle_id_25()
    # customer_vehicle_id = get_customer_vehicle_id(id=vehicle_id)
    # customer_vehicle_id = get_customer_vehicle_id(id=vehicle_id)

    experiment_start_time = settings.experiment_start_time
    experiment_end_time = settings.experiment_start_time + settings.experiment_last_time
    task_by_time_list = []
    fixed_distance_matrix_list = []
    mobile_distance_matrix_list = []


    # pool = multiprocessing.Pool(processes=5)
    # jobs = []
    
    result = process_wrapper(customer_vehicle_id, fixed_edge_node)
    task_by_time_list = result["task_by_time"]
    fixed_distance_matrix_list = result["fixed_distance_matrix"]
    # mobile_distance_matrix_list.append(result["mobile_distance_matrix"])

    # results.sort(key=lambda result: result["time"])
    #
    # for result in results:
    #     task_by_time_list.append(result["task_by_time"])
    #     fixed_distance_matrix_list.append(result["fixed_distance_matrix"])
    #     # mobile_distance_matrix_list.append(result["mobile_distance_matrix"])

    if save_pickle(fixed_edge_node,
                   task_by_time_list,
                   fixed_distance_matrix_list):
                   #mobile_distance_matrix_list):
        print_to_console("实验设置配置存储成功")