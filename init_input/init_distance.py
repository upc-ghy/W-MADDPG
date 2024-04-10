from init_input.init_edge_node import init_fixed_edge_node, init_edge_vehicle_node
from config.config import settings
from init_input.init_task_by_time import init_task_by_time
from init_input.init_vehicles import get_vehicle_id, get_customer_vehicle_id, get_edge_vehicle_id
import numpy as np


def get_distance_between_two_nodes(x1, y1, x2, y2):
    """根据2点的坐标计算欧式距离
    :argument
        x1  点一的横坐标，float
        y1  点一的纵坐标，float
        x2  点二的横坐标，float
        y2  点二的纵坐标，float
    :return
        distance    两点间的距离
    """
    distance = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return distance


def get_fixed_distance_matrix(fixed_edge_node, task_list):
    """得到固定边缘节点与客户车辆的距离矩阵
    :argument
        fixed_edge_node     固定边缘节点，List(Dict)
        task_list           任务列表，List(Dict)
    :return
        fixed_distance_matrix   固定边缘节点与客户车辆的距离矩阵
    Example
        fixed_distance_matrix[i][j] 表示固定边缘节点i与任务j之间的距离
    """
    # print("fixed_edge_node")
    # print(fixed_edge_node)
    # print("task_list")
    # print(task_list)
    edge_node_length = len(fixed_edge_node)
    task_length = len(task_list)
    # print("edge_node_length")
    # print(edge_node_length)
    # print("task_length")
    # print(task_length)

    fixed_distance_matrix = np.zeros((edge_node_length, task_length))
    for i in range(edge_node_length):
        x1 = fixed_edge_node[i]["x"]
        y1 = fixed_edge_node[i]["y"]
        # print("x1")
        # print(x1)
        # print(y1)
        for j in range(task_length):
            x2 = task_list[j]["x"]
            # print(x2)
            y2 = task_list[j]["y"]
            # print(y2)
            # print("y2")
            fixed_distance_matrix[i][j] = get_distance_between_two_nodes(x1, y1, x2, y2)
    return fixed_distance_matrix


def get_mobile_distance_matrix(time, edge_vehicle_node, task_list):
    """得到移动边缘节点与客户车辆的距离矩阵
    :argument
        time                时间，int, 由于边缘车辆节点的位置不固定，与时间有关
        edge_vehicle_node   移动边缘节点，List(Dict)
        task_list           任务列表，List(Dict)
    :return
        mobile_distance_matrix       固定边缘节点与客户车辆的距离矩阵
    Example
        mobile_distance_matrix[i][j] 表示移动边缘节点i与任务j之间的距离
    """
    edge_vehicle_length = len(edge_vehicle_node)
    task_length = len(task_list)
    time_no = time - settings.experiment_start_time
    mobile_distance_matrix = np.zeros((edge_vehicle_length, task_length))
    for i in range(edge_vehicle_length):
        x1 = edge_vehicle_node[i]["x_list"][time_no]
        y1 = edge_vehicle_node[i]["y_list"][time_no]
        for j in range(task_length):
            x2 = task_list[j]["x"]
            y2 = task_list[j]["y"]
            mobile_distance_matrix[i][j] = get_distance_between_two_nodes(x1, y1, x2, y2)
    return mobile_distance_matrix


def get_task_id_under_edge_node(node_type, node_id, distance_matrix):
    """得到边缘节点node_id 覆盖范围内的任务id
    :argument
        node_type           节点的类型，可选值["BaseStation","RSU","Vehicle"]
        node_id             节点的ID, int
        distance_matrix     距离矩阵，matrix, 根据节点类型注意选择对应的距离矩阵
    :return
        task_id             任务id, List
    """
    task_id_list_all = [[26, 25, 27], [4, 0, 2, 5, 3, 1], [15, 18, 16, 19, 17], [21, 20, 24, 23, 22],
                        [13, 9, 14, 7, 8, 6], [10, 12, 11]]
    task_id = []
    if node_type == "BaseStation":
        return task_id_list_all[node_id]
    elif node_type == "RSU":
        return task_id_list_all[node_id]
    else:
        raise ValueError("from init_distance.get_task_list_under_edge_node 节点类型出错， 不是指定的类型")

def get_task_id_under_edge_node_10(node_type, node_id, distance_matrix):
    """得到边缘节点node_id 覆盖范围内的任务id
    :argument
        node_type           节点的类型，可选值["BaseStation","RSU","Vehicle"]
        node_id             节点的ID, int
        distance_matrix     距离矩阵，matrix, 根据节点类型注意选择对应的距离矩阵
    :return
        task_id             任务id, List
    """
    task_id_list_all = [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3]]
    task_id = []
    if node_type == "BaseStation":
        return task_id_list_all[node_id]
    elif node_type == "RSU":
        return task_id_list_all[node_id]
    else:
        raise ValueError("from init_distance.get_task_list_under_edge_node 节点类型出错， 不是指定的类型")

def get_task_id_under_edge_node_15(node_type, node_id, distance_matrix):
    """得到边缘节点node_id 覆盖范围内的任务id
    :argument
        node_type           节点的类型，可选值["BaseStation","RSU","Vehicle"]
        node_id             节点的ID, int
        distance_matrix     距离矩阵，matrix, 根据节点类型注意选择对应的距离矩阵
    :return
        task_id             任务id, List
    """
    task_id_list_all = [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14]]
    task_id = []
    if node_type == "BaseStation":
        return task_id_list_all[node_id]
    elif node_type == "RSU":
        return task_id_list_all[node_id]
    else:
        raise ValueError("from init_distance.get_task_list_under_edge_node 节点类型出错， 不是指定的类型")
def get_task_id_under_edge_node_20(node_type, node_id, distance_matrix):
    """得到边缘节点node_id 覆盖范围内的任务id
    :argument
        node_type           节点的类型，可选值["BaseStation","RSU","Vehicle"]
        node_id             节点的ID, int
        distance_matrix     距离矩阵，matrix, 根据节点类型注意选择对应的距离矩阵
    :return
        task_id             任务id, List
    """
    task_id_list_all = [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    task_id = []
    if node_type == "BaseStation":
        return task_id_list_all[node_id]
    elif node_type == "RSU":
        return task_id_list_all[node_id]
    else:
        raise ValueError("from init_distance.get_task_list_under_edge_node 节点类型出错， 不是指定的类型")

def get_task_id_under_edge_node_25(node_type, node_id, distance_matrix):
    """得到边缘节点node_id 覆盖范围内的任务id
    :argument
        node_type           节点的类型，可选值["BaseStation","RSU","Vehicle"]
        node_id             节点的ID, int
        distance_matrix     距离矩阵，matrix, 根据节点类型注意选择对应的距离矩阵
    :return
        task_id             任务id, List
    """
    task_id_list_all = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 18], [10, 11, 14, 15, 16, 17], [9, 12, 13, 24, 19], [20, 21, 22, 23]]
    task_id = []
    if node_type == "BaseStation":
        return task_id_list_all[node_id]
    elif node_type == "RSU":
        return task_id_list_all[node_id]
    else:
        raise ValueError("from init_distance.get_task_list_under_edge_node 节点类型出错， 不是指定的类型")


def get_task_time_limitation_under_edge_node(iteration, node_type, node_id, distance_matrix_list, task_list):
    task_id_list = []
    # task_id_list = get_task_id_under_edge_node(node_type, node_id, distance_matrix_list)
    task_id_list_all = [[26, 25, 27], [4, 0, 2, 5, 3, 1], [15, 18, 16, 19, 17], [21, 20, 24, 23, 22], [13, 9, 14, 7, 8, 6], [10, 12, 11]]
    task_id_list = task_id_list_all[node_id]
    # print("task_list")
    # print(task_list)
    # print("task_id_list")
    # print(task_id_list)
    task_time_limitation = np.zeros(len(task_id_list))
    for i in range(len(task_time_limitation)):
        task_id = task_id_list[i]
        task = task_list[task_id]
        task_time_limitation[i] = task["deadline"]
    return task_time_limitation


def get_task_time_limitation_under_edge_node_10(iteration, node_type, node_id, distance_matrix_list, task_list):
    task_id_list = []
    # print("node_id")
    # print(node_id)
    # task_id_list = get_task_id_under_edge_node(node_type, node_id, distance_matrix_list)
    task_id_list_all = [[4, 5, 6, 7, 8, 9], [0, 1, 2, 3]]
    task_id_list = task_id_list_all[node_id]
    # print("task_list")
    # print(task_list)
    # print("task_id_list")
    # print(task_id_list)
    task_time_limitation = np.zeros(len(task_id_list))
    for i in range(len(task_id_list)):
        task_id = task_id_list[i]
        task = task_list[task_id]
        task_time_limitation[i] = task["deadline"]
        # print("123")
    return task_time_limitation
    # return

def get_task_time_limitation_under_edge_node_15(iteration, node_type, node_id, distance_matrix_list, task_list):
    task_id_list = []
    # print("node_id")
    # print(node_id)
    # task_id_list = get_task_id_under_edge_node(node_type, node_id, distance_matrix_list)
    task_id_list_all = task_id_list_all = [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14]]
    task_id_list = task_id_list_all[node_id]
    # print("task_list")
    # print(task_list)
    # print("task_id_list")
    # print(task_id_list)
    task_time_limitation = np.zeros(len(task_id_list))
    for i in range(len(task_id_list)):
        task_id = task_id_list[i]
        task = task_list[task_id]
        task_time_limitation[i] = task["deadline"]
        # print("123")
    return task_time_limitation

def get_task_time_limitation_under_edge_node_20(iteration, node_type, node_id, distance_matrix_list, task_list):
    task_id_list = []
    # print("node_id")
    # print(node_id)
    # task_id_list = get_task_id_under_edge_node(node_type, node_id, distance_matrix_list)
    task_id_list_all = [[4, 5, 6, 7, 9, 10], [0, 1, 2, 3], [8, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    task_id_list = task_id_list_all[node_id]
    # print("task_list")
    # print(task_list)
    # print("task_id_list")
    # print(task_id_list)
    task_time_limitation = np.zeros(len(task_id_list))
    for i in range(len(task_id_list)):
        task_id = task_id_list[i]
        task = task_list[task_id]
        task_time_limitation[i] = task["deadline"]
        # print("123")
    return task_time_limitation

def get_task_time_limitation_under_edge_node_25(iteration, node_type, node_id, distance_matrix_list, task_list):
    task_id_list = []
    # print("node_id")
    # print(node_id)
    # task_id_list = get_task_id_under_edge_node(node_type, node_id, distance_matrix_list)
    task_id_list_all = [[0, 1, 2, 3, 4, 5], [6, 7, 8], [10, 11, 14, 15, 16, 17, 18, 19], [9, 12, 13, 24],
                        [20, 21, 22, 23]]
    task_id_list = task_id_list_all[node_id]
    # print("task_list")
    # print(task_list)
    # print("task_id_list")
    # print(task_id_list)
    task_time_limitation = np.zeros(len(task_id_list))
    for i in range(len(task_id_list)):
        task_id = task_id_list[i]
        task = task_list[task_id]
        task_time_limitation[i] = task["deadline"]
        # print("123")
    return task_time_limitation

if __name__ == '__main__':
    time = 1

    fixed_edge_node = init_fixed_edge_node()
    edge_vehicle_id = get_edge_vehicle_id()
    id = get_vehicle_id()
    customer_vehicle_id = get_customer_vehicle_id(edge_vehicle_id, id)
    task_list = init_task_by_time(customer_vehicle_id, time)
    edge_vehicle_num = settings.edge_vehicle_num
    edge_vehicle_node = init_edge_vehicle_node(edge_vehicle_num=edge_vehicle_num,
                                               edge_vehicle_id=edge_vehicle_id)

    fixed_distance_matrix = get_fixed_distance_matrix(fixed_edge_node, task_list)
    mobile_distance_matrix = get_mobile_distance_matrix(time, edge_vehicle_node, task_list)
    print("*" * 32)
    print("Fixed Distance Matrix")
    print(fixed_distance_matrix)
    print("*" * 32)
    print("Mobile Distance Matrix")
    print(mobile_distance_matrix)

    print("*" * 32)
    print("Task Under Edge Node")
    node_type = "BaseStation"
    print(node_type)
    task_id = get_task_id_under_edge_node(node_type=node_type,
                                          node_id=0,
                                          distance_matrix=fixed_distance_matrix)
    print(task_id)
    node_type = "RSU"
    print(node_type)
    task_id = get_task_id_under_edge_node(node_type=node_type,
                                          node_id=7,
                                          distance_matrix=fixed_distance_matrix)
    print(task_id)
    node_type = "Vehicle"
    print(node_type)
    task_id = get_task_id_under_edge_node(node_type=node_type,
                                          node_id=0,
                                          distance_matrix=mobile_distance_matrix)
    print(task_id)
    # distance = get_distance_between_two_nodes(1, 1, 1, 1)
    # print(distance)
    # print(type(distance))
    # matrix = np.zeros((2,2))
    # print(matrix)
    # matrix[1][1] = 2
    # print(matrix)
