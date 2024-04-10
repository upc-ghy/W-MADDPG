import numpy as np
import tensorflow as tf

import Environment_marl_3
from model_agent_maddpg_3 import MADDPG
from replay_buffer_3 import ReplayBuffer
import pandas as pd
import csv
from pathlib import Path
import pickle
import copy


def create_init_update(oneline_name, target_name, tau=0.99):
    online_var = [i for i in tf.trainable_variables() if oneline_name in i.name]
    target_var = [i for i in tf.trainable_variables() if target_name in i.name]

    target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
    target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in zip(online_var, target_var)]

    return target_init, target_update


agent1_ddpg = MADDPG('agent1')
agent1_ddpg_target = MADDPG('agent1_target')

agent2_ddpg = MADDPG('agent2')
agent2_ddpg_target = MADDPG('agent2_target')

agent3_ddpg = MADDPG('agent3')
agent3_ddpg_target = MADDPG('agent3_target')

agent4_ddpg = MADDPG('agent4')
agent4_ddpg_target = MADDPG('agent4_target')

agent5_ddpg = MADDPG('agent5')
agent5_ddpg_target = MADDPG('agent5_target')

saver = tf.train.Saver()

# saver.save(sess,"")

agent1_actor_target_init, agent1_actor_target_update = create_init_update('agent1_actor', 'agent1_target_actor')
agent1_critic_target_init, agent1_critic_target_update = create_init_update('agent1_critic', 'agent1_target_critic')

agent2_actor_target_init, agent2_actor_target_update = create_init_update('agent2_actor', 'agent2_target_actor')
agent2_critic_target_init, agent2_critic_target_update = create_init_update('agent2_critic', 'agent2_target_critic')

agent3_actor_target_init, agent3_actor_target_update = create_init_update('agent3_actor', 'agent3_target_actor')
agent3_critic_target_init, agent3_critic_target_update = create_init_update('agent3_critic', 'agent3_target_critic')

agent4_actor_target_init, agent4_actor_target_update = create_init_update('agent4_actor', 'agent4_target_actor')
agent4_critic_target_init, agent4_critic_target_update = create_init_update('agent4_critic', 'agent4_target_critic')

agent5_actor_target_init, agent5_actor_target_update = create_init_update('agent5_actor', 'agent5_target_actor')
agent5_critic_target_init, agent5_critic_target_update = create_init_update('agent5_critic', 'agent5_target_critic')


def get_agents_action(o_n, sess, i_episode, saver):
    if i_episode < 2000:
        var = 1 - (1 - 0.1) * i_episode/ 2000  # epsilon decreases over each episode
    elif (i_episode > 2000) & i_episode < 10000:
        var = 0.1
    else:
        var = 0
    i = np.random.random(1)
        # print("var的值为：")
        # print(var)
        # print("i的值为：")
        # print(i)
    if(i >= var):
        agent1_action = agent1_ddpg.action("../experiment_output/agent1_ddpg", i_episode, saver, state=[o_n[0]], sess=sess)
        agent2_action = agent2_ddpg.action("../experiment_output/agent2_ddpg", i_episode, saver, state=[o_n[1]], sess=sess)
        agent3_action = agent3_ddpg.action("../experiment_output/agent3_ddpg", i_episode, saver, state=[o_n[2]], sess=sess)
        agent4_action = agent4_ddpg.action("../experiment_output/agent4_ddpg", i_episode, saver, state=[o_n[3]], sess=sess)
        agent5_action = agent5_ddpg.action("../experiment_output/agent5_ddpg", i_episode, saver, state=[o_n[4]], sess=sess)
        # print("----------------agent1_action---------------")
        # print(agent1_action)
        # agent5_action = agent5_ddpg.action("agent5_ddpg", i_episode, saver, state=[o_n[4]], sess=sess) + np.random.randn(2) * 0.5
    else:
        # agent1_action = agent1_ddpg.action("model_256/agent1_ddpg", i_episode, saver, state=[o_n[0]], sess=sess)
        # agent2_action = agent2_ddpg.action("model_256/agent2_ddpg", i_episode, saver, state=[o_n[1]], sess=sess)
        # agent3_action = agent3_ddpg.action("model_256/agent3_ddpg", i_episode, saver, state=[o_n[2]], sess=sess)
        # agent4_action = agent4_ddpg.action("model_256/agent4_ddpg", i_episode, saver, state=[o_n[3]], sess=sess)
        # agent5_action = agent5_ddpg.action("model_256/agent5_ddpg", i_episode, saver, state=[o_n[4]], sess=sess)
        # print("----------------agent1_action---------------")
        # print(agent1_action)
        
        agent1_action = [np.random.uniform(-1, 1, size=2)]
        agent2_action = [np.random.uniform(-1, 1, size=2)]
        agent3_action = [np.random.uniform(-1, 1, size=2)]
        agent4_action = [np.random.uniform(-1, 1, size=2)]
        agent5_action = [np.random.uniform(-1, 1, size=2)]
    # agent1_action = [[-0.12209407  0.19717381]]
    return agent1_action, agent2_action, agent3_action, agent4_action, agent5_action

def train_agent(agent_ddpg_name, i_episode, agent_ddpg, agent_ddpg_target, agent_memory, agent_actor_target_update, agent_critic_target_update, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch = agent_memory.sample(256)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :], total_act_batch[:, 3, :],total_act_batch[:, 4, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]
    next_other_actor2_o = total_next_obs_batch[:, 2, :]
    next_other_actor3_o = total_next_obs_batch[:, 3, :]
    next_other_actor4_o = total_next_obs_batch[:, 4, :]
    # 获取下一个情况下另外三个agent的行动
    next_other_action = np.hstack([other_actors[0].action("agent_1", 0, saver, next_other_actor1_o, sess), other_actors[1].action("agent_1", 0, saver, next_other_actor2_o, sess), other_actors[2].action("agent_1", 0, saver, next_other_actor3_o, sess), other_actors[3].action("agent_1", 0, saver, next_other_actor4_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.9999 * agent_ddpg_target.Q(state=next_obs_batch, action=agent_ddpg_target.action("agent_1", 0, saver, next_obs_batch, sess),
                                                                     other_action=next_other_action, sess=sess)
    agent_ddpg.train_actor(agent_ddpg_name, i_episode, saver, agent_ddpg, state=obs_batch, other_action=other_act_batch, sess=sess)
    agent_ddpg.train_critic(state=obs_batch, action=act_batch, other_action=other_act_batch, target=target, sess=sess)

    sess.run([agent_actor_target_update, agent_critic_target_update])
    # saver.save(sess,"actor")

def get_state(env, i, ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    # include V2I/V2V fast_fading, V2V interference, V2I/V2V 信道信息（PL+shadow）,
    # 剩余时间, 剩余负载
    # @修改当前函数，设置当前基站的状态为[基站下任务的数据量大小、任务剩余时延、每个信道所受干扰]
    task_on_base = env.task_on_base
    task_data_size = env.task_data_size
    task_deadline = env.task_deadline
    task_state = []
    for j in range(len(task_on_base[i])):
        task_state.append((task_data_size[task_on_base[i][j]]-30000.0)/30000.0)
    if len(task_state) < 10:
        for j in range(10 - len(task_state)):
            task_state.append((0-30000.0)/30000.0)
    for j in range(len(task_on_base[i])):
        task_state.append((task_deadline[task_on_base[i][j]]-1500)/1500.0)
    if len(task_state) < 20:
        for j in range(20 - len(task_state)):
            task_state.append((0-1500)/1500.0)
    # print("task_state")
    # print(task_state)

    channel_infer = env.V2V_Interference_all[i]

    # print("channel_infer")
    # print(channel_infer)
    # V2I_fast = [0.15779032 0.33851582 0.53765134 0.68601726]
    return np.concatenate((task_state, channel_infer))
    # 这里有所有感兴趣的物理量：V2V_fast V2I_fast V2V_interference V2I_abs V2V_abs、剩余时间、剩余负载

def load_experiment_median_from_pickle(input_number):
    json_file = Path('../experiment_data/experiment_median_file_name.txt')
    with json_file.open('r', encoding="utf-8") as fp:
        file_lines = fp.readlines()
        file_line = file_lines[input_number - 1]
        file_name = str(file_line).replace('\n', '')
    pickle_file = Path(file_name)
    if pickle_file.exists():
        return file_name
    else:
        raise FileNotFoundError("from init_input.experiment_input_save_and_reload Pickle File not found")


def canshu_init():
    df = pd.read_csv('../data/test_data.csv')
    task_x = df['x']
    task_y = df['y']
    task_length = len(task_x)

    # 任务的数据量大小

    pickle_file = Path(load_experiment_median_from_pickle(1))
    fp = pickle_file.open("rb")
    iteration = pickle.load(fp)
    fixed_edge_node = pickle.load(fp)
    fixed_distance_matrix = pickle.load(fp)
    task_list = pickle.load(fp)
    fixed_node_num = pickle.load(fp)
    max_potential_value = pickle.load(fp)
    useful_channel_under_node = pickle.load(fp)
    # print("useful_channel_under_node")
    # print(useful_channel_under_node)
    # print_to_console("useful_channel_under_node",useful_channel_under_node)
    task_id_under_each_node_list = pickle.load(fp)
    usable_channel_of_all_nodes = pickle.load(fp)
    task_time_limitation_of_all_nodes = pickle.load(fp)
    combination_and_strategy_length_of_all_nodes = pickle.load(fp)

    task_data_size = []
    task_deadline = []
    for i in range(len(task_list)):
        task_deadline.append(task_list[i]["deadline"])
        task_data_size.append(task_list[i]["data_size"])
    # print_to_console("task_data_size",task_data_size)

    # 每个基站下的用户
    task_on_base = task_id_under_each_node_list
    task_on_base_matrix = np.zeros((6, task_length))
    distance_matrix = np.zeros((6, task_length))
    for i in range(len(task_on_base)):
        for j in range(len(task_on_base[i])):
            task_on_base_matrix[i][task_on_base[i][j]] = 1

    # 时延和数据量矩阵
    task_state = []
    for i in range(len(task_on_base)):
        task_state_tmp = []
        for j in range(len(task_on_base[i])):
            task_state_tmp.append(task_data_size[task_on_base[i][j]])
        task_state.append(task_state_tmp)
    for i in range(len(task_on_base)):
        if len(task_state[i])<10:
            for j in range(10-len(task_on_base[i])):
                task_state[i].append(0)
    for i in range(len(task_on_base)):
        for j in range(len(task_on_base[i])):
            task_state[i].append(task_deadline[task_on_base[i][j]])

    for i in range(len(task_on_base)):
        if len(task_state[i]) < 20:
            for j in range(20 - len(task_on_base[i])):
                task_state[i].append(0)
    # print("task_state")
    # print(task_state)
    # print_to_console("task_data_size_base", task_data_size_base)
    # print_to_console("task_deadline_base", task_deadline_base)
    # new_task_data_size_base = copy.deepcopy(task_data_size_base)
    # print(task_data_size_base.shape)
    # print(task_deadline_base.shape)

    # 信道的空闲情况
    init_sub_channel = []
    channel_on_base_matrix = np.zeros((5, 10))
    for i in range(len(useful_channel_under_node)):
        init_sub_channel.append(useful_channel_under_node[i]['node_channel'])
        for j in useful_channel_under_node[i]['node_channel']:
            channel_on_base_matrix[i][j] = 1


    return task_on_base, task_on_base_matrix, fixed_distance_matrix, usable_channel_of_all_nodes, task_deadline, task_data_size


if __name__ == '__main__':
    task_on_base, task_on_base_matrix, fixed_distance_matrix, usable_channel_of_all_nodes, task_deadline, task_data_size = canshu_init()

    sim_dict_init = {'n_veh': 5,
                     'n_neighbor':1,
                     'sub_channel_bandwidth': 5e4,  # 子信道带宽 50Khz = 5*10^4 Hz
                     'MBS_transmission_power_max': 100,  # 宏基站最大传输功率 mW
                     'RSU_transmission_power_max': 20,  # 小基站最大传输功率 mW
                     'current_busy': 1e-8,  # 高斯白噪声
                     'MBS_gain': 8,
                     'RSU_gain': 2,
                     'dt': 20,
                     'N_agents': 5,
                     'PATH_LOSS_EXPONENT': 3.7,
                     'task_on_base': task_on_base,
                     'task_on_base_matrix': task_on_base_matrix,
                     'fixed_distance_matrix': fixed_distance_matrix,
                     'usable_channel_of_all_nodes': usable_channel_of_all_nodes,
                     'task_deadline': task_deadline,
                     'task_data_size': task_data_size}  # palyer的数量
    sim_dict = copy.deepcopy(sim_dict_init)
    n_RB = 5
    env = Environment_marl_3.Environ(sim_dict)

    # n_episode = 502
    n_episode = 20000
    # n_episode = 20000
    n_step_per_episode = 10
    # n_step_per_episode = int(env.time_slow / env.time_fast)  # 0.1/0.001 = 100
    epsi_final = 0.01  # 探索最终值          ##13
    epsi_anneal_length = int(0.8 * 3000)  # 探索退火长度
    mini_batch_step = n_step_per_episode     # 100
    target_update_step = n_step_per_episode * 4  # 400
    success_count = 0
    flag = 0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run([agent1_actor_target_init, agent1_critic_target_init,
              agent2_actor_target_init, agent2_critic_target_init,
              agent3_actor_target_init, agent3_critic_target_init,
              agent4_actor_target_init, agent4_critic_target_init,
              agent5_actor_target_init, agent5_critic_target_init])

    memory_size = 50000
    agent1_memory = ReplayBuffer(memory_size)
    agent2_memory = ReplayBuffer(memory_size)
    agent3_memory = ReplayBuffer(memory_size)
    agent4_memory = ReplayBuffer(memory_size)
    agent5_memory = ReplayBuffer(memory_size)

    df = open("action.csv", "w", newline="")
    csv_writer = csv.writer(df)
    df2 = open("sum_reward.csv","w",newline="")
    csv_writer2 = csv.writer(df2)

    for i_episode in range(n_episode):
        print("-------------------------")

        sim_dict = copy.deepcopy(sim_dict_init)
        # print("sim_dict")
        # print(sim_dict)
        env = Environment_marl_3.Environ(sim_dict)

        if i_episode < epsi_anneal_length:
            var = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
        elif (i_episode > epsi_anneal_length) & i_episode < 10000:
            var = epsi_final
        else:
            var = 0
    #     if i_episode % 100 == 0:      # 每100次更新一次位置、邻居、快衰、信道。
    #         env.renew_positions()   # update vehicle position
    #         env.renew_neighbor()
    #         env.renew_channel()
    #         env.renew_channels_fastfading()     # update channel fast fading
    #
        # 与邻近节点的时延和数据量大小
        sum_reward = 0
        epsi_V2I_Rate = 0
        epsi_V2V_Rate = 0
        epsi_V2V_success = 0
        i_step = 0
        fail_sum = 0

        while (not (np.any(env.task_deadline) == 0)) & (not (np.any(env.task_data_size) == 0)):
        # for i_step in range(n_step_per_episode):  # range内是0.1/0.001 = 100
            # 初始化state_old_all,action_all action_all_training
            # time_step = i_episode * n_step_per_episode + i_step  # time_step是整体的step
            state_old_all = []
            action_all = []
            states = []
            action_all_training = np.zeros([5, 1, 2])   ####11
            for i in range(5):  # 对每一个链路
                state = get_state(env, i, i_episode, var)   # 获取该链路的state【对于单个链路】
                # print("state")
                # print(state)
                states.append(state)
                state_old_all.append(state)
                # print(state_old_all)
            if success_count >= 10:
                # i_episode = 19999
                flag = 1
                agent1_action, agent2_action, agent3_action, agent4_action, agent5_action = np.clip(np.array(get_agents_action(states, sess, i_episode, saver)), -1, 1)  # 通过predict得到action（包含RB和POWER的信息）【对于单个链路】
            else:
                agent1_action, agent2_action, agent3_action, agent4_action, agent5_action = np.clip(np.array(get_agents_action(states, sess, i_episode, saver)), -1,1)  # 通过predict得到action（包含RB和POWER的信息）【对于单个链路】

            action = [agent1_action.tolist(), agent2_action.tolist(), agent3_action.tolist(), agent4_action.tolist(), agent5_action.tolist()]
            # print("action")
            # print(action)
            action_all.append(agent1_action)
            action_all.append(agent2_action)
            action_all.append(agent3_action)
            action_all.append(agent4_action)
            action_all.append(agent5_action)

            for i in range(5):  # 对每一个链路
                for j in range(1):
                    # print("-----------777action[i][j][0]----------------")
                    # print(action[i][j])
                    action_all_training[i, j, 0] = action[i][j][0]  # chosen RB
                    action_all_training[i, j, 1] = action[i][j][1]  # power level
            # print("action_all_training")
            # print(action_all_training)
    #
    #         # All agents take actions simultaneously, obtain shared reward, and update the environment.
            action_temp = action_all_training.copy()
            # print("action_temp")
            # print(action_temp)
            train_reward,success,fail = env.act_for_training(action_temp)  # 通过action_for_training得到reward【这里是对于所有链路的】如果是sarl，则把计算reward的放到上面的for内，其他一样
            sum_reward += train_reward
            fail_sum += fail
            # print("sum_reward")
            # print(sum_reward)
            # print(train_reward)

            epsi_V2V_success += success


            if i_episode < 10 or i_episode > n_episode - 100:
                print("Step: ", i_step)
                i_step += 1
                print("success: ", epsi_V2V_success)
                print("fail：", fail_sum)
    #
            # env.renew_channels_fastfading()  # 更新快衰
            # env.Compute_Interference(action_temp)  # 根据action计算干扰
    #
            state_new_all = []
            for i in range(5):              # 使用for循环对每个链路
                state_new = get_state(env, i, i_episode/(n_episode-1), var)   # 计算新状态
                state_new_all.append(state_new)
            # action_t = np.concatenate(np.asarray([i_episode, var, round(sum_reward, 4)]), np.concatenate(agent1_action[0],
            #                           agent2_action[0], agent3_action[0]),np.concatenate(agent4_action[0], agent5_action[0]))
            # action_t = np.concatenate(np.asarray([i_episode, var, round(sum_reward, 4)]),agent1_action[0])
            action_t = [i_episode, var, round(train_reward, 4)]
            action_t.extend(agent1_action[0])
            action_t.extend(agent2_action[0])
            action_t.extend(agent3_action[0])
            action_t.extend(agent4_action[0])
            action_t.extend(agent5_action[0])

            csv_writer.writerow(action_t)

            agent1_memory.add(np.vstack([state_old_all[0], state_old_all[1], state_old_all[2], state_old_all[3], state_old_all[4]]),
                              np.vstack([agent1_action[0], agent2_action[0], agent3_action[0], agent4_action[0], agent5_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[0], state_new_all[1], state_new_all[2], state_new_all[3], state_new_all[4]]))  # add entry to this agent's memory将（state_old,state_new,train_reward,action)加入agent的memory中【所以说这里的memory每一条是对于单个链路的】

            agent2_memory.add(np.vstack([state_old_all[1], state_old_all[2], state_old_all[3], state_old_all[4], state_old_all[0]]),
                              np.vstack([agent2_action[0], agent3_action[0], agent4_action[0], agent5_action[0], agent1_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[1], state_new_all[2], state_new_all[3], state_new_all[4], state_new_all[0]]))

            agent3_memory.add(np.vstack([state_old_all[2], state_old_all[3], state_old_all[4], state_old_all[0], state_old_all[1]]),
                              np.vstack([agent3_action[0], agent4_action[0], agent5_action[0], agent1_action[0], agent2_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[2], state_new_all[3], state_new_all[4], state_new_all[0], state_new_all[1]]))

            agent4_memory.add(np.vstack([state_old_all[3], state_old_all[4], state_old_all[0], state_old_all[1], state_old_all[2]]),
                              np.vstack([agent4_action[0], agent5_action[0], agent1_action[0], agent2_action[0], agent3_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[3], state_new_all[4], state_new_all[0], state_new_all[1], state_new_all[2]]))

            agent5_memory.add(np.vstack([state_old_all[4], state_old_all[0], state_old_all[1], state_old_all[2], state_old_all[3]]),
                              np.vstack([agent5_action[0], agent1_action[0], agent2_action[0], agent3_action[0], agent4_action[0]]),
                              train_reward,
                              np.vstack([state_new_all[4], state_new_all[0], state_new_all[1], state_new_all[2], state_new_all[3]]))

            if i_episode > 100:

                train_agent("agent1_ddpg",i_episode, agent1_ddpg, agent1_ddpg_target, agent1_memory, agent1_actor_target_update,
                            agent1_critic_target_update, sess, [agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target, agent5_ddpg_target])

                train_agent("agent2_ddpg", i_episode, agent2_ddpg, agent2_ddpg_target, agent2_memory, agent2_actor_target_update,
                            agent2_critic_target_update, sess, [agent3_ddpg_target, agent4_ddpg_target, agent5_ddpg_target, agent1_ddpg_target])

                train_agent("agent3_ddpg", i_episode, agent3_ddpg, agent3_ddpg_target, agent3_memory, agent3_actor_target_update,
                            agent3_critic_target_update, sess, [agent4_ddpg_target, agent5_ddpg_target, agent1_ddpg_target, agent2_ddpg_target])

                train_agent("agent4_ddpg", i_episode, agent4_ddpg, agent4_ddpg_target, agent4_memory, agent4_actor_target_update,
                            agent4_critic_target_update, sess, [agent5_ddpg_target, agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target])

                train_agent("agent5_ddpg", i_episode, agent5_ddpg, agent5_ddpg_target, agent5_memory, agent5_actor_target_update,
                            agent5_critic_target_update, sess,[agent1_ddpg_target, agent2_ddpg_target, agent3_ddpg_target, agent4_ddpg_target])

        #
        print("Episode：" + str(i_episode) + ", Explore：" + str(round(var, 4)) + ", Reward: " + str(round(sum_reward, 4)) + ",success：" + str(epsi_V2V_success) + ",fail：" + str(fail_sum))
        # if Episode> 500 &(epsi_V2V_success==10):
        #     break
        if i_episode > 500:
            if epsi_V2V_success >= 10:
                success_count += 1
            else:
                success_count = 0

        reward_csv = []
        reward_csv.append(i_episode)
        reward_csv.append(sum_reward)
        reward_csv.append(epsi_V2V_success)
        csv_writer2.writerow(reward_csv)

    df2.close()
    df.close()
    sess.close()
