from datetime import *
import pickle
import random
from config.config import settings
from pathlib import Path


def get_median_unique_file_name():
    now_time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    random_number = random.randint(0, 100)  # 生成随机数n,其中0<=n<=100
    if random_number < 10:
        random_number = str(0) + str(random_number)
    unique_file_name = "../experiment_data/experiment_median/" + "Experiment_median_" + "25.pkl"
    return unique_file_name


def save_experiment_median_to_pickle(iteration,
                                     fixed_edge_node,
                                     fixed_distance_matrix,
                                     task_list,
                                     node_num,
                                     max_potential_value,
                                     useful_channel_under_node,
                                     task_id_under_each_node_list,
                                     usable_channel_of_all_nodes,
                                     task_time_limitation_of_all_nodes,
                                     combination_and_strategy_length_of_all_nodes
                                     ):
    file_name = get_median_unique_file_name()
    txt_file = Path(settings.EXPERIMENT_MEDIAN_FILE_NAME)
    with txt_file.open('a+', encoding='utf-8') as fp:
        fp.write(file_name + "\n")
    pickle_file = Path(file_name)
    with pickle_file.open("wb") as fp:
        pickle.dump(iteration, fp)
        pickle.dump(fixed_edge_node, fp)
        pickle.dump(fixed_distance_matrix, fp)
        pickle.dump(task_list, fp)
        pickle.dump(node_num, fp)
        print("&&&&&node_num")
        print(node_num)
        pickle.dump(max_potential_value, fp)
        pickle.dump(useful_channel_under_node, fp)
        pickle.dump(task_id_under_each_node_list, fp)
        pickle.dump(usable_channel_of_all_nodes, fp)
        pickle.dump(task_time_limitation_of_all_nodes, fp)
        pickle.dump(combination_and_strategy_length_of_all_nodes, fp)
        return True


def load_experiment_median_from_pickle(input_number):
    json_file = Path(settings.EXPERIMENT_MEDIAN_FILE_NAME)
    file_name = ""
    with json_file.open('r', encoding="utf-8") as fp:
        file_lines = fp.readlines()
        print(file_lines)
        file_line = file_lines[input_number - 1]
        file_name = str(file_line).replace('\n', '')
    print(file_name)
    pickle_file = Path(file_name)
    if pickle_file.exists():
        return file_name
    else:
        raise FileNotFoundError("from init_input.experiment_input_save_and_reload Pickle File not found")