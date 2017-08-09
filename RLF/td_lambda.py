import os,sys
import numpy as np

class Temporal_Difference_lambda:

    def __init__(self,state_list,action_list):

        self.states = state_list
        self.actions = action_list

        self.state_num = len(self.states)
        self.action_num = len(self.actions)

        self.Q = dict()
        for s in self.states:
            self.Q[s] = np.random.random(self.action_num)
            #self.Q[s] = np.zeros(self.action_num)

        self.Z = dict()
        for s in self.states:
            self.Z[s] = np.zeros(self.action_num)

    def reset_Z(self):

        for s in self.states:
            self.Z[s] = np.zeros(self.action_num)

    def set_policy(self,learning_type):

        self.pi = dict()

        if learning_type == 'sarsa_lambda':
            for s in self.states:
                self.pi[s] = np.random.random(self.action_num)
                self.pi[s] = self.pi[s] / np.sum(self.pi[s])
        else:
            pass


    def sarsa_lambda(self,agent,episode_num,epsilon,alpha,gamma,Lambda,max_timestep,eval_interval):

        ep_idx = 0
        avg_ep_return_list = []

        while ep_idx < episode_num:

            if ep_idx % eval_interval == 0:
                eval_ep = agent.episode_generator(self.pi, max_timestep, True)
                print("eval episode length:%d" % (len(eval_ep) / 3))
                c_avg_return = agent.avg_return_per_episode(eval_ep)
                avg_ep_return_list.append(c_avg_return)
                print("assessing return:%f" % c_avg_return)
                print "avg return list length:", len(avg_ep_return_list)

            ep_idx += 1
            print "ep_idx:",ep_idx

            self.reset_Z()

            agent.c_state = agent.getInitState()
            agent.next_state = agent.c_state

            c_action_idx = np.random.choice(self.action_num, 1, p=self.pi[agent.c_state])[0]
            #agent.c_action = self.actions[c_action_idx]


            n = 0

            while not (agent.isTerminated() or n >= max_timestep):

                agent.c_state = agent.next_state
                agent.c_action = self.actions[c_action_idx]

                agent.c_state, agent.c_action, agent.c_reward, agent.next_state = agent.oneStep_generator()

                next_action_idx = np.random.choice(self.action_num, 1, p=self.pi[agent.next_state])[0]

                delta = agent.c_reward + gamma * self.Q[agent.next_state][next_action_idx] - self.Q[agent.c_state][c_action_idx]

                self.Z[agent.c_state][c_action_idx] += 1

                for s in self.states:
                    for i in range(self.action_num):
                        self.Q[s][i] += alpha * delta * self.Z[s][i]
                        self.Z[s][i] *= gamma * Lambda

                # --------policy update at same time---------#
                c_best_action_idx = np.argmax(self.Q[agent.c_state])

                for action_idx in range(self.action_num):
                    if action_idx == c_best_action_idx:
                        self.pi[agent.c_state][action_idx] = 1 - epsilon + epsilon / self.action_num
                    else:
                        self.pi[agent.c_state][action_idx] = epsilon / self.action_num


                c_action_idx = next_action_idx
                n += 1

        return avg_ep_return_list

    def test_git(self):
        print "pull back..."
        pass

