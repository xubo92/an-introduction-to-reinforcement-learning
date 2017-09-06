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
        self.mu = dict()

        if learning_type == 'sarsa_lambda':
            for s in self.states:
                self.pi[s] = np.random.random(self.action_num)
                self.pi[s] = self.pi[s] / np.sum(self.pi[s])
        elif learning_type == 'naive_Q_lambda':
            for s in self.states:
                idx = np.random.randint(0, self.action_num, size=1)[0]
                self.pi[s] = np.zeros(self.action_num)
                self.pi[s][idx] = 1.0

                self.mu[s] = np.random.random(self.action_num)
                self.mu[s] = self.mu[s] / np.sum(self.mu[s])



    def sarsa_lambda(self,env,episode_num,epsilon,alpha,gamma,Lambda,max_timestep,eval_interval):

        ep_idx = 0
        avg_ep_return_list = []

        while ep_idx < episode_num:

            if ep_idx % eval_interval == 0:
                eval_ep = env.episode_generator(self.pi, max_timestep, True)
                print("eval episode length:%d" % (len(eval_ep) / 3))
                c_avg_return = env.avg_return_per_episode(eval_ep)
                avg_ep_return_list.append(c_avg_return)
                print("assessing return:%f" % c_avg_return)
                print "avg return list length:", len(avg_ep_return_list)

            ep_idx += 1
            print "ep_idx:",ep_idx

            self.reset_Z()

            env.c_state = env.getInitState()
            env.next_state = env.c_state

            c_action_idx = np.random.choice(self.action_num, 1, p=self.pi[env.c_state])[0]
            #env.c_action = self.actions[c_action_idx]


            n = 0

            while not (env.isTerminated() or n >= max_timestep):

                env.c_state = env.next_state
                env.c_action = self.actions[c_action_idx]

                env.c_state, env.c_action, env.c_reward, env.next_state = env.oneStep_generator()

                next_action_idx = np.random.choice(self.action_num, 1, p=self.pi[env.next_state])[0]

                delta = env.c_reward + gamma * self.Q[env.next_state][next_action_idx] - self.Q[env.c_state][c_action_idx]

                self.Z[env.c_state][c_action_idx] += 1

                for s in self.states:
                    for i in range(self.action_num):
                        self.Q[s][i] += alpha * delta * self.Z[s][i]
                        self.Z[s][i] *= gamma * Lambda

                # --------policy update at same time---------#
                c_best_action_idx = np.argmax(self.Q[env.c_state])

                for action_idx in range(self.action_num):
                    if action_idx == c_best_action_idx:
                        self.pi[env.c_state][action_idx] = 1 - epsilon + epsilon / self.action_num
                    else:
                        self.pi[env.c_state][action_idx] = epsilon / self.action_num


                c_action_idx = next_action_idx
                n += 1
                print "n:",n

        return avg_ep_return_list

    # to improve the performance of Watkin's Q(lambda) and reduce the complexity of Peng's Q(lambda),we introduce naive Q(lambda)
    def naive_Q_lambda(self,env,episode_num,epsilon,alpha,gamma,Lambda,max_timestep,eval_interval):

        ep_idx = 0
        avg_ep_return_list = []

        while ep_idx < episode_num:

            if ep_idx % eval_interval == 0:
                eval_ep = env.episode_generator(self.pi, max_timestep, True)
                print("eval episode length:%d" % (len(eval_ep) / 3))
                c_avg_return = env.avg_return_per_episode(eval_ep)
                avg_ep_return_list.append(c_avg_return)
                print("assessing return:%f" % c_avg_return)
                print "avg return list length:", len(avg_ep_return_list)

            ep_idx += 1
            print "ep_idx:", ep_idx

            self.reset_Z()

            env.c_state = env.getInitState()
            env.next_state = env.c_state

            c_action_idx = np.random.choice(self.action_num, 1, p=self.mu[env.c_state])[0]
            # env.c_action = self.actions[c_action_idx]


            n = 0

            while not (env.isTerminated() or n >= max_timestep):
                env.c_state = env.next_state
                env.c_action = self.actions[c_action_idx]

                env.c_state, env.c_action, env.c_reward, env.next_state = env.oneStep_generator()

                next_action_idx = np.random.choice(self.action_num, 1, p=self.mu[env.next_state])[0]
                next_best_action_idx = np.argmax(self.Q[env.next_state])

                delta = env.c_reward + gamma * self.Q[env.next_state][next_best_action_idx] - self.Q[env.c_state][next_action_idx]
                self.Z[env.c_state][env.c_action] += 1


                if next_action_idx == next_best_action_idx:
                    for s in self.states:
                        for i in range(self.action_num):
                            self.Q[s][i] += alpha * delta * self.Z[s][i]
                            self.Z[s][i] *= Lambda * gamma
                else:
                    for s in self.states:
                        for i in range(self.action_num):
                            self.Q[s][i] += alpha * delta * self.Z[s][i]
                            self.Z[s][i] = 0.5

                c_best_action_idx = np.argmax(self.Q[env.c_state])





                # ------- update behavior policy --------- #
                for action_idx in range(self.action_num):
                    if action_idx == c_best_action_idx:
                        self.mu[env.c_state][action_idx] = 1 - epsilon + epsilon / self.action_num
                    else:
                        self.mu[env.c_state][action_idx] = epsilon / self.action_num

                # --------target policy update at same time---------#
                for action_idx in range(self.action_num):
                    if action_idx == c_best_action_idx:
                        self.pi[env.c_state][action_idx] = 1.0
                    else:
                        self.pi[env.c_state][action_idx] = 0.0

                c_action_idx = next_action_idx
                n += 1

        return avg_ep_return_list