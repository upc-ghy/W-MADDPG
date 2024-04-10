import tensorflow as tf
import tensorflow.contrib as tc

class MADDPG():
    def __init__(self, name, layer_norm=True, nb_actions=2, nb_input=26, nb_other_aciton=8):
        gamma = 0.999
        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32,name = 'x')
        # action_output = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32, name='output_node')
        # print(state_input)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_aciton], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # state_input_2 = tf.placeholder(shape=[None, nb_input], dtype=tf.float32, name='x')


        def actor_network(name):
            # with tf.variable_scope(name) as scope:
            x = state_input
            # x = tf.layers.dense(x, 64)
            x = tf.layers.dense(x, 256)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)       # 第一层

            # x = tf.layers.dense(x, 64)
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)       # 第二层

            # x = tf.layers.dense(x, 64)
            x = tf.layers.dense(x, 16)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)  # 第三层

            x = tf.layers.dense(x, self.nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=0))
            x = tf.nn.tanh(x,name = "y")       # 第四层
                # print("xxxx")
                # print(x)
            return x

        def critic_network(name, action_input, reuse=False):
            with tf.variable_scope(name) as scope:
                if reuse:
                    scope.reuse_variables()

                # # x = state_input     #14
                x = tf.concat([state_input, action_input], axis=-1)       #14

                x = tf.layers.dense(x, 256)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                # x = tf.concat([x, action_input], axis=-1)       #14
                # x = tf.layers.dense(x, 64)
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 16)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=0, maxval=3e-3))
            return x

        self.action_output = actor_network(name + "_actor")
        self.critic_output = critic_network(name + '_critic',
                                            action_input=tf.concat([action_input, other_action_input], axis=1))
        self.state_input = state_input
        self.action_input = action_input
        self.other_action_input = other_action_input
        self.reward = reward

        self.actor_optimizer = tf.train.AdamOptimizer(1e-3)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)

        global_steps = tf.contrib.framework.get_or_create_global_step()
        # learning_rate = tf.train.exponential_decay(0.001,global_steps,10,2,staircase=False)
        # learning_rate = 1e-3
        # self.actor_optimizer = tf.train.AdagradOptimizer(learning_rate)
        # self.critic_optimizer = tf.train.AdagradOptimizer(learning_rate)

        # 最大化Q值
        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic', action_input=tf.concat([self.action_output, other_action_input], axis=1),
                           reuse=True))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss, global_step=global_steps)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = -tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss, global_step=global_steps)

    def train_actor(self, agent_ddpg_name, i_episode, server, agent_ddpg, state, other_action, sess):
        sess.run(self.actor_train, {self.state_input: state, self.other_action_input: other_action})
        # if i_episode == 596:
        #     server.save(sess, agent_ddpg_name +"_actor_4_1.ckpt")
                # print(str(agent_ddpg))


    def train_critic(self, state, action, other_action, target, sess):
        sess.run(self.critic_train,
                 {self.state_input: state, self.action_input: action, self.other_action_input: other_action,
                  self.target_Q: target})

    def action(self, ddpg_name, i_episode, saver, state, sess):
        if i_episode == 19999:
            saver.save(sess, ddpg_name + "_actor.ckpt")
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action, self.other_action_input: other_action})