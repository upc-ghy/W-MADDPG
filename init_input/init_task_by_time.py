from config.config import settings
from init_input.init_vehicles import get_vehicle_location, get_customer_vehicle_id, get_edge_vehicle_id, get_vehicle_id
import random
from tqdm import tqdm
import pandas as pd


def get_random_task_data_size():
    """从任务数据大小中随机生成一个数据大小
    :argument
        task_data_size_min      数据大小最小值，int
        task_data_size_max      数据大小最大值，int
    :return
        random_task_data_size   数据大小
    """
    random_task_data_size = random.randint(settings.task_data_size_min, settings.task_data_size_max)
    return random_task_data_size


def get_random_task_deadline():
    """从任务截止时间范围值中随机选择一个截止时间
    :argument
        task_deadline_min   截止时间最小值, int
        task_deadline_max   截止时间最大值, int
    :return
        random_task_deadline    截止时间，int
    """
    random_task_data_size = random.randint(settings.TASK_DEADLINE_MIN, settings.TASK_DEADLINE_MAX)
    return random_task_data_size


def init_task_by_time(customer_vehicle_id):
    """得到time时间的所有任务
    :argument
        customer_vehicle_id     客户车辆ID,int
        time                    时间，int
    :return
        task_list               任务列表，List
    """
    df = pd.read_csv(settings.FILE_XY_DEL)
    vehicle_id_x = df["x"].tolist()
    vehicle_id_y = df["y"].tolist()
    # print("vehicle_id_y")
    # print(vehicle_id_y)
    # print(vehicle_id_x)
    task_list = []
    # tqdm 进度条
    for id in customer_vehicle_id:
        # vehicle is not none
        task = {"task_id": id,
                "x": vehicle_id_x[id-1],
                "y": vehicle_id_y[id-1],
                "data_size": get_random_task_data_size(),
                "deadline": get_random_task_deadline()}
        task_list.append(task)
    return task_list
def init_task_by_time_10(customer_vehicle_id):
    """得到time时间的所有任务
    :argument
        customer_vehicle_id     客户车辆ID,int
        time                    时间，int
    :return
        task_list               任务列表，List
    """
    df = pd.read_csv(settings.TEST_DATA_DEL_10)
    vehicle_id_x = df["x"].tolist()
    vehicle_id_y = df["y"].tolist()
    # print("vehicle_id_y")
    # print(vehicle_id_y)
    # print(vehicle_id_x)
    task_list = []
    # tqdm 进度条
    for id in customer_vehicle_id:
        # vehicle is not none
        task = {"task_id": id,
                "x": vehicle_id_x[id-1],
                "y": vehicle_id_y[id-1],
                "data_size": get_random_task_data_size(),
                "deadline": get_random_task_deadline()}
        task_list.append(task)
    return task_list

def init_task_by_time_15(customer_vehicle_id):
    """得到time时间的所有任务
    :argument
        customer_vehicle_id     客户车辆ID,int
        time                    时间，int
    :return
        task_list               任务列表，List
    """
    df = pd.read_csv(settings.TEST_DATA_DEL_15)
    vehicle_id_x = df["x"].tolist()
    vehicle_id_y = df["y"].tolist()
    # print("vehicle_id_y")
    # print(vehicle_id_y)
    # print(vehicle_id_x)
    task_list = []
    # tqdm 进度条
    for id in customer_vehicle_id:
        # vehicle is not none
        task = {"task_id": id,
                "x": vehicle_id_x[id-1],
                "y": vehicle_id_y[id-1],
                "data_size": get_random_task_data_size(),
                "deadline": get_random_task_deadline()}
        task_list.append(task)
    return task_list

def init_task_by_time_20(customer_vehicle_id):
    """得到time时间的所有任务
    :argument
        customer_vehicle_id     客户车辆ID,int
        time                    时间，int
    :return
        task_list               任务列表，List
    """
    df = pd.read_csv(settings.TEST_DATA_DEL_20)
    vehicle_id_x = df["x"].tolist()
    vehicle_id_y = df["y"].tolist()
    # print("vehicle_id_y")
    # print(vehicle_id_y)
    # print(vehicle_id_x)
    task_list = []
    # tqdm 进度条
    for id in customer_vehicle_id:
        # vehicle is not none
        task = {"task_id": id,
                "x": vehicle_id_x[id-1],
                "y": vehicle_id_y[id-1],
                "data_size": get_random_task_data_size(),
                "deadline": get_random_task_deadline()}
        task_list.append(task)
    return task_list

def init_task_by_time_25(customer_vehicle_id):
    """得到time时间的所有任务
    :argument
        customer_vehicle_id     客户车辆ID,int
        time                    时间，int
    :return
        task_list               任务列表，List
    """
    df = pd.read_csv(settings.TEST_DATA_DEL_25)
    vehicle_id_x = df["x"].tolist()
    vehicle_id_y = df["y"].tolist()
    # print("vehicle_id_y")
    # print(vehicle_id_y)
    # print(vehicle_id_x)
    task_list = []
    # tqdm 进度条
    for id in customer_vehicle_id:
        # vehicle is not none
        task = {"task_id": id,
                "x": vehicle_id_x[id-1],
                "y": vehicle_id_y[id-1],
                "data_size": get_random_task_data_size(),
                "deadline": get_random_task_deadline()}
        task_list.append(task)
    return task_list

if __name__ == '__main__':
    # edge_vehicle_id = get_edge_vehicle_id()
    # id = get_vehicle_id()
    # customer_vehicle_id = get_customer_vehicle_id(edge_vehicle_id=edge_vehicle_id,
    #                                               id=id)
    # task_list = init_task_by_time(customer_vehicle_id=customer_vehicle_id, time=2)
    # print("*" * 32)
    # print("Task List")
    # print(task_list)
    # print("*" * 32)
    # print("Task List Example")
    # print(task_list[0])
    print(get_random_task_deadline())