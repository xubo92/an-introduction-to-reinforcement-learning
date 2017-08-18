import os,sys
import numpy as np

class Dyna_Q:

    def __init__(self,state_list,action_list):

        self.states = state_list
        self.actions = action_list

        self.state_num  = len(self.states)
        self.action_num = len(self.actions)

        self.Q = dict()
        for s in self.states:
            self.Q[s] = np.zeros(self.action_num)

        self.Model = dict()

        for s in self.states:
            self.Model[s] = list()
            for i in range(self.action_num):
                rand_state = self.states[np.randint(0,self.state_num-1)]
                self.Model[s].append([0,rand_state])

    def set_policy(self,learning_type):

        self.pi = dict()

        if learning_type == 'Dyna-Q':
            for s in self.states:
                self.pi[s] = np.random.random(self.action_num)
                self.pi[s] = self.pi[s] / np.sum(self.pi[s])

    def Dyna_Q_learning(self,agent,episode_num,epsilon,alpha,gamma,max_timestep,planning_num,eval_interval):

        ep_idx = 0

        avg_ep_return_list = []

        observed_sa = dict()

        while ep_idx < episode_num:

            ep_idx += 1

            agent.c_state = agent.getInitState()
            agent.next_state = agent.c_state

            n = 0

            c_action_idx = np.random.choice(self.action_num, 1, p=self.pi[agent.c_state])[0]
            agent.c_action = self.actions[c_action_idx]

            while not (agent.isTerminated() or n >= max_timestep) :

                agent.c_state = agent.next_state
                agent.c_action = self.actions[c_action_idx]

                if agent.c_state in observed_sa.keys():
                    observed_sa[agent.c_state].append(c_action_idx)
                else:
                    observed_sa[agent.c_state] = [c_action_idx]

                agent.c_state, agent.c_action, agent.c_reward, agent.next_state = agent.oneStep_generator()

                next_action_idx = np.random.choice(self.action_num, 1, p=self.pi[agent.next_state])[0]

                self.Q[agent.c_state][c_action_idx] += alpha * (agent.c_reward + gamma * self.Q[agent.next_state][next_action_idx] - self.Q[agent.c_state][c_action_idx])

                self.Model[agent.c_state][c_action_idx] = [agent.c_reward,agent.next_state]

                for plan_idx in range(planning_num):
                    pass


