# An intelligent task offloading method based on multi-agent deep reinforcement learning in ultra-dense heterogeneous network with mobile edge computing

# Requirements
'''
python 3.7
tensorflow
numpy
pandas
'''

# Abstract

With the rapid development of IoT technology, various computation-intensive and latency-sensitive tasks have emerged in large numbers, which impose higher requirements on the processing efficiency of the tasks. In this paper, by fusing mobile edge computing (MEC) and ultra-dense heterogeneous network (UD-HetNet) technologies, we design the scenario of UD-HetNet with MEC to improve the efficiency of computation and communication during task processing. However, how to design a effective task offloading and resource management strategy is still a major challenge in this scenario. Therefore, in this paper, we propose a distributed task offloading and wireless resource management framework that optimizes task offloading, local computation frequency scaling, subchannel allocation, and transmit power regulation strategies to reduce system overhead (weighted sum of delay and energy consumption of task processing) effectively. First, we design a task offloading for multi-base station (BS) collaboration base on priority algorithm and optimize the local computation frequency using convex optimization theory. Following by, we introduce a multi-agent deep deterministic policy gradient (MADDPG) technology to optimize subchannel allocation and transmit power regulation during task offloading to accommodate the dynamic and variable nature of the channel. Finally, user equipment (UE)-edge server (ES), UE-subchannel, and subchannel-power matching are achieved. Simulation results show that our algorithm has significant advantages in balancing the ES load, improving channel utilization, and reducing system overhead.

![image](https://github.com/upc-ghy/W-MADDPG/assets/133858812/f2ec5919-7b56-4308-9bd3-45d541807f6a)


![image](https://github.com/upc-ghy/W-MADDPG/assets/133858812/328a3281-d682-490f-bb59-1ae104b751bd)


# Contact information of some authors:

**Shanchen Pang:** pangsc@upc.edu.cn

**Teng Wang:** s21070024@s.upc.edu.cn

**Haiyuan Gui:** guihaiyuan736570@gmail.com
