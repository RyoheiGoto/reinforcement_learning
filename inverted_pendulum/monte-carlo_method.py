import matplotlib.pyplot as plt
import numpy as np
from plot import Penplot

##############################################
cart_mass = 0.5
pole_mass = 0.1
pole_length = 0.25
pole_masslength = pole_mass * pole_length
total_mass = cart_mass + pole_mass

gravity = 9.8
force_mag = 4.5
tau = 0.02

four_thirds = 1.3333333333333

one_degrees = 0.0174532
six_degrees = 0.1047192
twelve_degrees = 0.2094384
eighteen_degrees = 0.3141592653589793
fifty_degrees = 0.87266

epsilon = 0.1
gamma = 0.4
##############################################

class Pendulum(object):
    def __init__(self):
        print "-" * 30
        np.random.seed()
        self.Q = np.zeros([2, 3, 3, 6, 3])

    def update_status(self, old_state, action):
        x, x_dot, theta, theta_dot = old_state
        
        force = force_mag if action > 0 else -force_mag
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        temp = (force + pole_masslength * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (gravity * sin_theta - cos_theta * temp) / (pole_length * (four_thirds - pole_mass * cos_theta**2 / total_mass))
        x_acc = temp - pole_masslength * theta_acc * cos_theta / total_mass
        
        x += tau * x_dot
        x_dot += tau * x_acc
        theta += tau * theta_dot
        theta_dot += tau * theta_acc
        
        return x, x_dot, theta, theta_dot

    def e_greedy(self, state, act=2):
        policy = []
        x, x_dot, theta, theta_dot = self.threshold(state)
        quantity = [self.Q[action, x, x_dot, theta, theta_dot] for action in xrange(act)]
        
        for action in quantity:
            if action == max(quantity):
                policy.append(1. - epsilon + epsilon / act)
            else:
                policy.append(epsilon / act)
        
        if sum(policy) == 1.:
            return policy
        else:
            return [1.0 / act for i in xrange(act)]

    def decide_action(self, policy):
        prob = 0.
        for act in xrange(len(policy)):
            prob += policy[act]
            if np.random.random() < prob:
                return act
    
    def get_reward(self, state):
        x, x_dot, theta, theta_dot = [np.fabs(val) for val in state]
        
        if x > 2.4 or theta > twelve_degrees:
            return 0.
        else:
            return 1.

    def threshold(self, state):
        x, x_dot, theta, theta_dot = state
        s1 = s2 = s3 = s4 = 0
        
        if x < -0.8:
            s1 = 0
        elif x < 0.8:
            s1 = 1
        else:
            s1 = 2
        
        if x_dot < -0.5:
            s2 = 0
        elif x_dot < 0.5:
            s2 = 1
        else:
            s2 = 2
        
        if theta < -six_degrees:
            s3 = 0
        elif theta < -one_degrees:
            s3 = 1
        elif theta < 0.:
            s3 = 2
        elif theta < one_degrees:
            s3 = 3
        elif theta < six_degrees:
            s3 = 4
        else:
            s3 = 5
        
        if theta_dot < -fifty_degrees:
            s4 = 0
        elif theta_dot < fifty_degrees:
            s4 = 1
        else:
            s4 = 2
        
        return s1, s2, s3, s4

    def step(self):
        state = self.s
        policy = self.e_greedy(state)
        action = self.decide_action(policy)
        
        new_state = self.update_status(state, action)
        reward = self.get_reward(new_state)
        
        self.s = new_state
        self.d.append((state, action, reward, new_state))
        
        return False if reward < 1.0 else True

    def mc(self):
        self.Q = np.zeros([2, 3, 3, 6, 3])
        num = np.zeros([2, 3, 3, 6, 3])
        
        for m in self.D:
            for t in xrange(len(m)):
                state, action, reward, new_state = m[t]
                x, x_dot, theta, theta_dot = self.threshold(state)
                d = m[t:]
                q = 0.0
                
                for current in xrange(len(d)):
                    q += gamma ** current * d[current][2]
                    
                self.Q[action, x, x_dot, theta, theta_dot] += q
                num[action, x, x_dot, theta, theta_dot] += 1.0
        
        num[num == 0.] = 1.0
        self.Q /= num

    def process(self, max_episode=1000, min_step=0, update=True, plot=False, save=None):
        self.D = []
        total_step = 0
        max_step = 0
        state_list = None
        
        for episode in np.arange(0, max_episode):
            self.d = []
            self.s = (0, 0, 0, 0)
            for step in np.arange(0, 100000):
                if not self.step():
                    total_step += step
                    max_step = max(max_step, step)
                    if max_step is step:
                        state_list = self.d
                    if step > min_step:
                        self.D.append(self.d)
                    break
            else:
                print "episode %d is complite %d steps" % (episode, 100000)
        
        print "Episode:\t%d" % max_episode
        print "Average:\t%d (%.2lfs)" % (total_step / max_episode, (total_step / max_episode) * tau)
        print "Max step:\t%d (%.2lfs)" % (max_step, max_step * tau)
        
        if plot and state_list:
            state = [d[0] for d in state_list]
            Penplot(state, anime=True, fig=False)
        if update:
            self.mc()
        if save:
            q, s = save.split(" ")
            state = [d[0] for d in state_list]
            np.save(q, self.Q)
            np.save(s, state)
        print "-" * 30

if __name__ == '__main__':
    pendulum = Pendulum()
    pendulum.process(10000, 0, plot=True)
    pendulum.process(10000, 0, plot=True)
    pendulum.process(10000, 0, update=False, plot=True)

