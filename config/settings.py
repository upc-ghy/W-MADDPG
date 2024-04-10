FILL_XY_CSV_NAME = "../data/fill_xy.csv"
TEST_DATA_DEL_25 = "../data/test_data.csv"


"""##################################
#  experiment  settings
##################################"""

"""##################################
#  edge node  settings
##################################"""
# 基站、rsu数量以及 子信道总数量
BASE_STATION_NUM = 1
RSU_NUM = 5
SUB_CHANNEL_NUM = 10

# 基站位置
BASE_STATION_X = [1000]
BASE_STATION_Y = [1000]

# rsu位置
RSU_X = [1590, 1379, 1062, 547, 1760]
RSU_Y = [1490, 1225, 1530, 820, 764]
RSU_X_10 = [1700, 1000]
RSU_Y_10 = [1000, 1500]
RSU_X_15 = [1700, 1000, 1096]
RSU_Y_15 = [1000, 1500, 533]

RSU_X_20 = [1700, 1000, 1096, 1774]
RSU_Y_20 = [1000, 1500, 533, 2098]

RSU_X_25 = [1590, 896, 1760, 882, 1188]
RSU_Y_25 = [1490, 1500, 764, 830, 272]

RSU_NUM_10 = 2
RSU_NUM_15 = 3
RSU_NUM_20 = 4
RSU_NUM_25 = 5
# RSU_X = [1000, 1500, 2500, 2500, 2500, 500, 1300, 500, 1700]
# RSU_Y = [1500, 500, 500, 1300, 2500, 2000, 2500, 2500, 1000]

# 通讯半径
BASE_STATION_COMMUNICATION_RADIUS = 1000
RSU_COMMUNICATION_RADIUS = 500
EDGE_VEHICLE_COMMUNICATION_RADIUS = 300

# 最大传输功率
BASE_STATION_TRANSMISSION_POWER_MAX = 400
RSU_TRANSMISSION_POWER_MAX = 400
EDGE_VEHICLE_TRANSMISSION_POWER_MAX = 1

# 边缘节点信道数量
BASE_STATION_SUB_CHANNEL_NUM = 10
RSU_SUB_CHANNEL_NUM = 10
EDGE_VEHICLE_SUB_CHANNEL_NUM = 10

# 子信道带宽 50Khz = 5*10^4 Hz
SUB_CHANNEL_BANDWIDTH = 5e4

"""##################################
#  vehicular transmission task settings
##################################"""
# 传输数据任务数据量向大小, 单位Mb
TASK_DATA_SIZE_MIN = 1e5
TASK_DATA_SIZE_MAX = 5e5

# 传输数据任务截止时间
TASK_DEADLINE_MIN = 1000
TASK_DEADLINE_MAX = 2000

"""##################################
#  wireless communication parameters value settings
##################################"""
# 信道衰落增益
CHANNEL_FADING_GAIN_EX = 8
CHANNEL_FADING_GAIN_DX = 0.4

# 无线相关常数
ANTENNA_CONSTANT = 1

# 路径衰落指数
PATH_LOSS_EXPONENT = 3.6

# 高斯白噪声
WHITE_GAUSSIAN_NOISE = 1e-8

"""##################################
#  algorithm parameters value settings
##################################"""
# 学习率
LEARNING_RATE = 10

"""##################################
#  experiment parameters value settings
##################################"""
# 实验开始时间
EXPERIMENT_START_TIME = 1

# 实验持续时间
EXPERIMENT_LAST_TIME = 10

EXPERIMENT_FILE_NAME = "../experiment_data/experiment_file_name.txt"
EXPERIMENT_MEDIAN_FILE_NAME = "../experiment_data/experiment_median_file_name.txt"
ITERATION_MEDIAN_FILE_NAME = "../experiment_data/iteration_median_file_name.txt"

# 边缘节点的类型
NODE_TYPE_BASE_STATION = "BaseStation"
NODE_TYPE_RSU = "RSU"
NODE_TYPE_VEHICLE = "Vehicle"

NODE_TYPE_FIXED = "Fixed_Edge_Node"
NODE_TYPE_MOBILE = "Mobile_Edge_Node"