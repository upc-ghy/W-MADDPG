from __future__ import division
import numpy as np
import time
import random
import math
import copy


np.random.seed(1)
'''
定义了架构的四个基本CLASS，分别是：V2Vchannels，V2Ichannels，Vehicle，Environ。
其中Environ的方法（即函数）最多，Vehicle没有函数只有几个属性，其余两者各有两个方法（分别是计算路损和阴影衰落）。
'''

def print_to_console(msg, objective=None):
    print("*" * 32)
    print(msg)
    if objective is not None:
        print(objective)
        print(type(objective))


class Environ:
    # 初始化需要传入4个list（为上下左右路口的位置数据）：down_lane, up_lane, left_lane, right_lane；地图的宽和高；
    # 车辆数和邻居数。除以上所提外，内部含有好多参数，如下：
    def __init__(self, sim_dict_init):
        self.bandwidth = int(5e4)
        self.sig2 = 1e-8
        self.task_on_base = sim_dict_init['task_on_base']
        self.task_on_base_matrix = sim_dict_init['task_on_base_matrix']
        self.fixed_distance_matrix = sim_dict_init['fixed_distance_matrix']
        self.usable_channel_of_all_nodes = sim_dict_init['usable_channel_of_all_nodes']
        self.task_deadline = sim_dict_init['task_deadline']
        self.task_data_size = sim_dict_init['task_data_size']
        # print("self.task_deadline")
        # print(self.task_deadline)
        # print("self.task_data_size")
        # print(self.task_data_size)
        self.power_choose = [0, 100, 200, 300, 400]
        self.V2V_Interference_all = np.zeros((5, 6)) + self.sig2
        self.task_length = 25
        self.PATH_LOSS_EXPONENT = 3.7
        self.t = 0
        self.overhead = 0
        self.total_data = 0
        self.total_time = 0


    # 计算干扰：Compute_Interference(self, actions)，通过+=的方法计算V2V_Interference_all
    def Compute_Interference(self, actions):
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_RB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0].astype(int)
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        # interference from V2I links
        for i in range(self.n_RB):
            for k in range(len(self.vehicles)):
                for m in range(len(channel_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer V2V links
        for i in range(len(self.vehicles)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k, :])):
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        # V2V_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                        V2V_Interference[k, m, channel_selection[i, j]] += power_selection[i, j] * 10 ** ((30 - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)

    # 执行训练：输入actions，通过Compute_Performance_Reward_Train计算最终reward
    def act_for_training(self, actions):
        channel_strategy = []
        power_startegy = []
        action_temp = actions.copy()
        for i in range(len(action_temp)):
            action_temp[i][0][0] = int((math.pow(len(self.task_on_base[i]) + 1, len(self.usable_channel_of_all_nodes[i]))-1)*((action_temp[i][0][0]+1)/2))
            channel_strategy.append(action_temp[i][0][0])
            action_temp[i][0][1] = int((math.pow(len(self.power_choose),len(self.task_on_base[i]))-1)* ((action_temp[i][0][1]+1)/2))
            power_startegy.append(action_temp[i][0][1])
        # print("action_temp")
        # print(action_temp)
        reward_elements,success,fail = self.step(channel_strategy,power_startegy)
        return reward_elements,success,fail

    def get_channel_choose(self, channel_index, task_index, channel_strategy):
        strategy_node = []
        while True:
            quotient = channel_strategy % (len(task_index) + 1)  # 余数
            # print_to_console("quotient", quotient)
            # print_to_console("random_num1", random_num)
            channel_strategy = channel_strategy // (len(task_index) + 1)  # 商
            # print_to_console("random_num2", random_num)
            strategy_node.append(quotient)
            if channel_strategy == 0:
                break
        # print_to_console("strategy_node", strategy_node)
        # strategy_node.reverse()
        if len(strategy_node) < len(channel_index):
            for i in range(len(channel_index) - len(strategy_node)):
                strategy_node.append(0)
                # print_to_console("有")
        strategy_node.reverse()
        # print_to_console("constructor_of_strategy", constructor_of_strategy)
        # print_to_console("channel_choose_all_node", channel_choose_all_node)
        # print_to_console("strategy_node", strategy_node)
        for i in range(len(strategy_node)):
            if strategy_node[i] == 0:
                strategy_node[i] = -1
            else:
                strategy_node[i] = task_index[int(strategy_node[i] - 1)]
                # strategy_node[i] = task_index[strategy_node[i]-1]
        # print_to_console("strategy_node",strategy_node)
        return strategy_node

    def get_power_choose(self, task_index, power_strategy):
        if len(task_index) == 0:
            return []
        if power_strategy < 0:
            power_strategy = 0

        strategy_mgt = []
        # print_to_console("power_strategy", power_strategy)
        while True:
            quotient = int(power_strategy % len(self.power_choose))  # 余数
            # print_to_console("quotient", quotient)
            # print_to_console("random_num1", random_num)
            power_strategy = power_strategy // len(self.power_choose)  # 商
            # print_to_console("random_num2", random_num)
            strategy_mgt.append(quotient)
            if power_strategy == 0:
                break
            # print_to_console("strategy_node", strategy_node)
            # strategy_node.reverse()
        if len(strategy_mgt) < len(task_index):
            # print_to_console("yyyyy")
            # print_to_console("startegy_mgt", strategy_mgt)
            for i in range(len(task_index) - len(strategy_mgt)):
                strategy_mgt.append(0)
                # print_to_console("有")
        strategy_mgt.reverse()
        # print_to_console("constructor_of_strategy", constructor_of_strategy)
        # print_to_console("channel_choose_all_node", channel_choose_all_node)
        # print_to_console("strategy_mgt", strategy_mgt)
        strategy_mgt_test = np.zeros(len(task_index))
        for i in range(len(strategy_mgt)):
            strategy_mgt_test[i] = self.power_choose[strategy_mgt[i]]
        # print_to_console("strategy_mgt_test",strategy_mgt_test)
        return strategy_mgt_test

    def Compute_Interference_2(self,distance_matrix, channel_index, strategy_channel_allocation, power_every_task):
        # print("power_every_task")
        # print(power_every_task)
        Infer_every_channel_tmp = np.zeros((5,10))
        Infer_every_channel = [[],[],[],[],[]]
        for i in range(len(channel_index)):
            for index,j in enumerate(channel_index[i]):
                for m in range(len(channel_index)):
                    if m != i:
                        for index_2, n in enumerate(channel_index[m]):
                            if n == j:
                                Infer_every_channel_tmp[i][j] += math.pow(8,2)*power_every_task[strategy_channel_allocation[m][index_2]]*math.pow(distance_matrix[i][strategy_channel_allocation[m][index_2]], 1-self.PATH_LOSS_EXPONENT)
                                # Infer_every_channel_tmp[i][j] += math.pow(8, 2)
        for i in range(len(channel_index)):
            for j in channel_index[i]:
                Infer_every_channel[i].append(Infer_every_channel_tmp[i][j])
        # print("Infer_every_channel")
        # print(Infer_every_channel)
        # print("self.sig2*np.ones(5,6)")
        # print(self.sig2*np.ones((5,6)))
        self.V2V_Interference_all = Infer_every_channel + self.sig2*np.ones((5,6))

    def get_inference(self, distance_matrix, channel_index, strategy_channel_allocation, i, task_no, power_every_task):
        task_on_channel = []
        for j in range(len(strategy_channel_allocation[i])):
            if task_no == strategy_channel_allocation[i][j]:
                task_on_channel.append(channel_index[i][j])
        # print_to_console("get inference task_on_channel", task_on_channel)
        # print_to_console("strategy_channel_allocation",strategy_channel_allocation)
        # print_to_console("len(channel_index", len(channel_index))
        # 计算每个信道所受到的干扰
        inf_sum = []
        for t in range(len(task_on_channel)):
            inf = 0
            for n in range(len(channel_index)):
                if n != i:
                    if task_on_channel[t] in channel_index[n]:
                        for j in range(len(channel_index[n])):
                            if channel_index[n][j] == task_on_channel[t]:
                                task_tmp = strategy_channel_allocation[n][j]
                                if (task_tmp != -1) & (task_tmp != -2):
                                    task_power = power_every_task[task_tmp]
                                    if distance_matrix[i][task_tmp] < 500:
                                        inf += task_power * math.pow(8, 2) * math.pow(500,
                                                                                  0 - self.PATH_LOSS_EXPONENT)
                                    else:
                                        inf += task_power * math.pow(8, 2) * math.pow(distance_matrix[i][task_tmp],
                                                                                  0 - self.PATH_LOSS_EXPONENT)
            inf_sum.append(inf)
        return inf_sum, task_on_channel

    def renew_task_on_base(self):
        # print_to_console("self.task_on_base_renew_1",self.task_on_base)
        task_tmp = [[],[],[],[],[]]
        for i in range(len(self.task_on_base)):
            for j in self.task_on_base[i]:
                if j != -1:
                    task_tmp[i].append(j)
        self.task_on_base = task_tmp
        # print_to_console("self.task_on_base_renew_2",self.task_on_base)

    def get_current_reward(self, new_state, last_state):
        # print_to_console("new_state",new_state.sum_task[1])
        # print_to_console("last_state",last_state.sum_task[1])
        # data_size_translation = np.zeros(self.N)
        data_size_translation = 0
        for i in range(len(new_state)):
                # 归一化
            data_size_translation += (last_state[i] - new_state[i]) / 5e5
        # print_to_console("data_size_translation[i]",data_size_translation[i])
        # print_to_console("data_size_translation", data_size_translation)
        return data_size_translation

    def step(self, channel_strategy, power_strategy):
        strategy_channel_allocation = []
        strategy_power_mgt = []
        task_to_channel = []
        task_length = self.task_length
        task_to_base = np.zeros(task_length)
        last_data_size = self.task_data_size.copy()
        total_reward = 0
        success = 0
        fail = 0
        punish = 0
        power_all = 0
        data_size_all = 0
        task_on_base_tmp = copy.deepcopy(self.task_on_base)
        # inf = np.zeros(5,6)

        task_channel = []
        for i in range(task_length):
            task_channel_tmp = []
            task_channel.append(task_channel_tmp)

        # 合并信道选择策略和功率控制策略
        action = []
        action.append(channel_strategy)
        action.append(power_strategy)

        if (not (np.any(self.task_deadline) == 0)) & (not (np.any(self.task_data_size) == 0)):
            # print_to_console("self.task_deadline",self.task_deadline)
            # print_to_console("self.task_data_size",self.task_data_size)
            for i in range(task_length):
                tmp = []
                task_to_channel.append(tmp)
            # 记录每一个任务位于哪个基站
            # print_to_console("self.task_on_base", self.task_on_base)
            for i in range(len(task_on_base_tmp)):
                for j in task_on_base_tmp[i]:
                    task_to_base[j] = i

            for i in range(len(channel_strategy)):
                # 得到每个任务的信道分配以及发射功率
                strategy_channel_allocation.append(
                    self.get_channel_choose(self.usable_channel_of_all_nodes[i], self.task_on_base[i], channel_strategy[i]))
                strategy_power_mgt.append(self.get_power_choose(self.task_on_base[i], power_strategy[i]))
            # print_to_console("channel_index", channel_index)
            # print_to_console("strategy_channel_allocation", strategy_channel_allocation)
            # print_to_console("strategy_power_mgt", strategy_power_mgt)
            # print_to_console("self.task_on_base_strategy",self.task_on_base)
            # [array([400., 400., 400.]), array([0., 0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0.]), array([0., 0., 0., 0., 0., 0.]), array([400., 400., 400.])]
            for i in range(len(self.usable_channel_of_all_nodes)):
                for j in range(len(self.usable_channel_of_all_nodes[i])):
                    if strategy_channel_allocation[i][j] != -1:
                        task_channel[strategy_channel_allocation[i][j]].append(self.usable_channel_of_all_nodes[i][j])

            new_channel_task_matrix = np.zeros((task_length, 10))  # 记录信道和任务的匹配关系
            # 记录信道和任务的匹配关系
            # print_to_console("strategy_channel_allocation",strategy_channel_allocation)
            # print_to_console("self.channel_index",self.channel_index)
            # print_to_console("self.channel_index", self.channel_index)
            for i in range(len(self.usable_channel_of_all_nodes)):
                for j in range(len(self.usable_channel_of_all_nodes[i])):
                    if strategy_channel_allocation[i][j] != -1:
                        new_channel_task_matrix[strategy_channel_allocation[i][j]][self.usable_channel_of_all_nodes[i][j]] = 1
            # strategy_channel_allocation
            power_every_task = np.zeros(self.task_length)
            for i in range(len(strategy_power_mgt)):
                for j in range(len(strategy_power_mgt[i])):
                    # print_to_console("strategy_power_mgt[i][j]",strategy_power_mgt[i][j])
                    power_every_task[task_on_base_tmp[i][j]] = strategy_power_mgt[i][j]
            # print_to_console("power_every_task",power_every_task)
            # 求每一个任务的信噪比以及传输的数据量
            S = np.zeros(self.task_length)
            v = np.zeros(self.task_length)
            # print_to_console("self.task_on_base", self.task_on_base)
            self.Compute_Interference_2(self.fixed_distance_matrix, self.usable_channel_of_all_nodes,strategy_channel_allocation, power_every_task)
            for i in range(len(task_on_base_tmp)):
                for j in range(len(task_on_base_tmp[i])):
                    if task_on_base_tmp[i][j] not in strategy_channel_allocation[i]:
                        S[task_on_base_tmp[i][j]] = 0
                    else:
                        # print_to_console("self.task_on_base[i][j]", self.task_on_base[i][j])
                        if self.fixed_distance_matrix[i][self.task_on_base[i][j]] < 500:
                            S[task_on_base_tmp[i][j]] = strategy_power_mgt[i][j] * math.pow(8, 2) * math.pow(
                                self.fixed_distance_matrix[i][task_on_base_tmp[i][j]], 0 - self.PATH_LOSS_EXPONENT)
                        else:
                            S[task_on_base_tmp[i][j]] = strategy_power_mgt[i][j] * math.pow(8, 2) * math.pow(
                                500, 0 - self.PATH_LOSS_EXPONENT)
                        # print_to_console("total_reward_power", total_reward)
                        total_reward -= strategy_power_mgt[i][j] * 200 * len(task_channel[task_on_base_tmp[i][j]])/ 5e5

                        # self.overhead += 2
                        power_all += strategy_power_mgt[i][j] * 200 * len(task_channel[task_on_base_tmp[i][j]])/ 5e5
                        # print_to_console("strategy_power_mgt[i][j] * 200 * len(task_channel[self.task_on_base[i][j]])/ 1e6", strategy_power_mgt[i][j] * 200 * len(task_channel[self.task_on_base[i][j]])/ 1e6)
                        # print_to_console("total_reward_power_l", total_reward)

                        if S[task_on_base_tmp[i][j]] != 0:

                            self.overhead += 0.2
                            self.overhead += strategy_power_mgt[i][j] * 200 * len(task_channel[task_on_base_tmp[i][j]]) / 1e5
                            # print_to_console("strategy_power_mgt[i][j] * 200 * len(task_channel[task_on_base_tmp[i][j]]) / 1e5",strategy_power_mgt[i][j] * 200 * len(task_channel[task_on_base_tmp[i][j]]) / 1e5)
                            # print_to_console("S[task_on_base[i][j]]", S[task_on_base[i][j]])
                            # 写一个函数：已知信道和基站求其他基站的任务对这个基站产生的干扰
                            inf, task_on_channel = self.get_inference(self.fixed_distance_matrix, self.usable_channel_of_all_nodes,
                                                                 strategy_channel_allocation, i, self.task_on_base[i][j], power_every_task)
                            # self.V2V_Interference_all[i][task_on_channel] = inf
                            # print_to_console("inf", inf)
                            # print_to_console("task_on_channel", task_on_channel)
                            # print_to_console("inf", inf)
                            # inf = get_inference(distance_matrix, channel_index, strategy_channel_allocation, i, 18, power_every_task)
                            for t in range(len(inf)):
                                # print_to_console("task_on_channel", task_on_channel)
                                # print_to_console("task_on_base[i][j]", task_on_base[i][j])
                                task_to_channel[task_on_base_tmp[i][j]] = task_on_channel
                                # print_to_console(task_to_channel)
                                # tmp = task_on_channel[1]
                                SINR = S[task_on_base_tmp[i][j]] / (inf[t] + self.sig2)
                                # print_to_console("SINR",SINR)
                                # self.total_time += 200
                                if SINR > 2:
                                    self.total_time += 200

                                    # print_to_console(">>>2")
                                    # print_to_console("SINR", SINR)
                                    tmp_task_data = 0
                                    v[task_on_base_tmp[i][j]] += round(self.bandwidth * np.log2(1 + SINR), 3)
                                    # new_task_deadline[i][task_on_base[i][j]] = 0
                                    # 判断一下与上一步是否使用的是同一个信道
                                    # sum_task[2]表示新的任务数据大小 new_task_data
                                    # print_to_console("self.channel_task_matrix",self.channel_task_matrix)
                                    # if (not (np.any(self.channel_task_matrix[task_on_base[i][j]]) == 0)):
                            # print("vv")
                            # print(v)
                            tmp_task_data = self.task_data_size[task_on_base_tmp[i][j]]
                            # print_to_console("self.task_data_size",self.task_data_size)
                            self.task_data_size[task_on_base_tmp[i][j]] = round((self.task_data_size[task_on_base_tmp[i][j]] - v[task_on_base_tmp[i][j]] * 0.2), 3)
                            # print_to_console("self.task_data_size",self.task_data_size)
                            # print_to_console("tmp_task_data", tmp_task_data)
                            if self.task_data_size[task_on_base_tmp[i][j]] <= 0 & (self.task_deadline[task_on_base_tmp[i][j]]/1000.0 - tmp_task_data/v[task_on_base_tmp[i][j]] >= 0):
                                self.task_data_size[task_on_base_tmp[i][j]] = 0
                                self.task_deadline[task_on_base_tmp[i][j]] = 0
                                # self.sum_task[4][i][self.task_on_base[i][j]] = 0
                                # self.sum_task[0][i][self.task_on_base[i][j]] = 0
                                # print_to_console("toward_power_loss",total_reward)
                                self.total_time -= 200 - (tmp_task_data/v[task_on_base_tmp[i][j]])*1000
                                tmp = (0.2 - tmp_task_data/v[task_on_base_tmp[i][j]]) * strategy_power_mgt[i][j]/500
                                total_reward += (0.2 - tmp_task_data/v[task_on_base_tmp[i][j]]) * strategy_power_mgt[i][j]/500
                                self.overhead -= (2 - tmp_task_data/v[task_on_base_tmp[i][j]]*10) + ((0.2 - tmp_task_data/v[task_on_base_tmp[i][j]]) * strategy_power_mgt[i][j]/100)
                                power_all -= tmp
                                # print_to_console("toward_power_loss_l", total_reward)
                                self.task_deadline[task_on_base_tmp[i][j]] = 200

                                # print_to_console("tmp",tmp)
                                # print_to_console("total_reward_no_success",total_reward)
                                # print_to_console("total_reward+5",total_reward)
                                total_reward += 5
                                # print_to_console("total_reward+5_l",total_reward)
                                # print_to_console("task_on_base_-1",self.task_on_base)
                                self.task_on_base[i][j] = -1
                                # print_to_console("task_on_base_-2",self.task_on_base)
                                success += 1
                            elif (self.task_data_size[task_on_base_tmp[i][j]] <= 0) & (
                                    self.task_deadline[task_on_base_tmp[i][j]]/1000.0 - tmp_task_data / v[task_on_base_tmp[i][j]] < 0):
                                self.task_data_size[task_on_base_tmp[i][j]] = 0
                                self.task_deadline[task_on_base_tmp[i][j]] = 200
                                fail += 1
                                self.task_on_base[i][j] = -1

                    self.task_deadline[task_on_base_tmp[i][j]] -= 200
                    # print_to_console("self.task_deadline",self.task_deadline)
                    # print_to_console("self.task_deadline[self.task_on_base[i][j]]", self.task_deadline[self.task_on_base[i][j]])
                    # print_to_console("1111111",self.sum_task[3][i][self.task_on_base[i][j]])


                    if (self.task_deadline[task_on_base_tmp[i][j]] < 0) & (self.task_data_size[task_on_base_tmp[i][j]] > 0):
                        # print_to_console("task_on_base_tmp[i][j]",task_on_base_tmp[i][j])
                        # print_to_console("self.task_deadline[task_on_base_tmp[i][j]]",self.task_deadline[task_on_base_tmp[i][j]])
                        # print_to_console("self.task_data_size[task_on_base_tmp[i][j]]",self.task_data_size[task_on_base_tmp[i][j]])
                        # print_to_console("self.task_on_base",self.task_on_base)
                        self.task_deadline[task_on_base_tmp[i][j]] = 0
                        punish += self.task_data_size[task_on_base_tmp[i][j]]
                        self.task_data_size[task_on_base_tmp[i][j]] = 0
                        total_reward -= 5
                        fail += 1
                        # print_to_console("fail")
                        self.task_on_base[i][j] = -1
                    # print_to_console("xian_task_deadline", task_deadline)
            self.channel_task_matrix = new_channel_task_matrix
            task_on_base_tmp = self.task_on_base.copy()
            self.renew_task_on_base()
            # print_to_console("self.get_current_reward(new_state, last_last_state)", self.get_current_reward(new_state, last_last_state))
            self.total_data += self.get_current_reward(self.task_data_size, last_data_size)*5e5
            # print_to_console("self.total_data",self.total_data)
            total_reward += self.get_current_reward(self.task_data_size, last_data_size) - punish/5e5
            data_size_all += self.get_current_reward(self.task_data_size, last_data_size) - punish/5e5
        return total_reward,success,fail


    # 执行测试：act_for_testing(self, actions)，这里和上面差不多，也用到了Compute_Performance_Reward_Train，
    # 但最后返回的是V2I_rate, V2V_success, V2V_rate。
    def act_for_testing(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements = self.Compute_Performance_Reward_Train(action_temp)
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates

        return V2I_Rate, V2V_success, V2V_Rate

    def act_for_testing_rand(self, actions):

        action_temp = actions.copy()
        V2I_Rate, V2V_Rate = self.Compute_Performance_Reward_Test_rand(action_temp)
        V2V_success = 1 - np.sum(self.active_links_rand) / (self.n_Veh * self.n_neighbor)  # V2V success rates

        return V2I_Rate, V2V_success, V2V_Rate
