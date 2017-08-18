import gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

class DQN:

    def __init__(self,env):

        self.env = env
        self.memory = list()

        self.gamma = 0.9
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.create_model()




    def create_model(self):

        model = Sequential()
        model.add(Dense(64,input_dim=4,activation='tanh'))
        model.add(Dense(128,activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse',optimizer = optimizers.RMSprop(lr=self.learning_rate))
        self.model = model

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def Deep_Q_Learning(self,env,episode_num,max_timestep,mini_batch_size,eval_interval):

        c_env = env
        ep_idx = 0

        for ep_idx in range(episode_num):

            c_state = c_env.reset()
            c_state = np.reshape(c_state,[1,4])

            n = 0

            for n in range(max_timestep):

                c_action_idx = self.act(c_state)
                next_state,c_reward,done,_= env.step(c_action_idx)
                next_state = np.reshape(next_state,[1,4])


                if done:
                    c_reward = -100
                else:
                    c_reward = c_reward

                self.remember(c_state, c_action_idx, c_reward, next_state, done)

                c_state = next_state

                if done:

                    print("episode: {}/{}, score: {}".format(ep_idx, episode_num, n))
                    break

                #--------------------------- start replay training -------------------------#

                batch_size = min(mini_batch_size,len(self.memory))
                batches_idx = np.random.choice(len(self.memory),batch_size)

                for i in batches_idx:
                    replay_c_state,replay_c_action_idx,replay_c_reward,replay_next_state,replay_done = self.memory[i]

                    if replay_done:
                        replay_c_target = replay_c_reward
                    else:
                        replay_c_target = replay_c_reward + self.gamma * np.amax(self.model.predict(replay_next_state)[0])

                    replay_c_expectValue = self.model.predict(replay_c_state)
                    replay_c_expectValue[0][replay_c_action_idx] = replay_c_target

                    self.model.fit(replay_c_state,replay_c_expectValue,nb_epoch=1, verbose=0)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

