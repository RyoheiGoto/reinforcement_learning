import numpy as np
import matplotlib.pyplot as plt
from plot import CarPlot, ValuePlot

class MountainCar(object):
    LEFT, NEUTRAL, RIGHT = range(3)

    def __init__(self):
        np.random.seed()
        
        #number of iterations
        self.L = 100
        #number of episodes
        self.M = 20
        #number of step
        self.T = 100
        
        #time constant
        self.tau = 0.1
        #coefficient of friction
        self.k = 0.3
        
        #weight[kg]
        self.m = 0.2
        #driving force[kgm/s^2]
        self.A = -0.2, 0, 0.2
        #position
        self.x = 0
        #acceleration
        self.dx = 0
        
        ###Gaussian function
        #centor vector
        self.c = np.array([ [-1.2, -1.5], [-1.2, -0.5], [-1.2, 0.5], [-1.2, 1.5],
                            [-0.35, -1.5], [-0.35, -0.5], [-0.35, 0.5], [-0.35, 1.5],
                            [0.5, -1.5], [0.5, -0.5], [0.5, 0.5], [0.5, 1.5 ]   ])
        #standard deviation
        self.sigma = 0.05
        
        ###temporal difference
        self.theta = np.zeros(36)
        self.gamma = 0.95
        self.X = np.ndarray(shape=(self.M * self.T, 36))
        self.X_num = 0
        self.reward = np.ndarray(shape=(self.M * self.T))
        self.reward_num = 0
        
        #e-greedy
        self.epsilon = 0.005

    def init_status(self):
        self.x = -0.5
        self.dx = 0
        self.mem_x = [self.x]
        
        self.X_num = 0
        self.reward_num = 0

    def decide_action(self, x, dx):
        policy = self.e_greedy(x, dx)
       
        tmp = 0
        for action, prob in enumerate(policy):
            tmp += prob
            if np.random.random() < tmp:
                return action
        
        return MountainCar.RIGHT

    def e_greedy(self, x, dx):
        s = np.array([x, dx])
        left = 0
        neutral = 0
        right = 0
        
        for i in range(12):
            left += self.theta[i] * np.exp(-(np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
            neutral += self.theta[i + 12] * np.exp(-(np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
            right += self.theta[i + 24] * np.exp(-(np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
        values = [left, neutral, right]
        
        if not sum(values):
            policy = [0.33, 0.33, 0.34]
        else:
            if max(values) == left:
                policy = [1.0 - self.epsilon + self.epsilon / 3.0, self.epsilon / 3.0, self.epsilon / 3.0]
            elif max(values) == neutral:
                policy = [self.epsilon / 3.0, 1.0 - self.epsilon + self.epsilon / 3.0, self.epsilon / 3.0]
            elif max(values) == right:
                policy = [self.epsilon / 3.0, self.epsilon / 3.0, 1.0 - self.epsilon + self.epsilon / 3.0]
        
        return policy

    def update_status(self, x, dx, action):
        dx += (-9.8 * self.m * np.cos(3.0 * x) + self.A[action] / self.m - self.k * dx) * self.tau
        x += dx * self.tau
        
        return x, dx

    def update_matrix(self, x, dx, action, reward):
        new_x, new_dx = self.update_status(x, dx, action)
        new_s = np.array([new_x, new_dx])
        old_s = np.array([x, dx])
        left = 0
        neutral = 0
        right = 0
        
        for i in range(12):
            left += self.theta[i] * np.exp(-(np.linalg.norm(new_s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
            neutral += self.theta[i + 12] * np.exp(-(np.linalg.norm(new_s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
            right += self.theta[i + 24] * np.exp(-(np.linalg.norm(new_s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
        values = [left, neutral, right]
        
        if not sum(values):
            new_action = np.random.randint(0, 3)
        else:
            new_action = values.index(max(values))
        
        for i in range(36):
            if action == MountainCar.LEFT and  0 <= i < 12:
                old_phi = np.exp(-(np.linalg.norm(old_s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
            elif action == MountainCar.NEUTRAL and 12 <= i < 24:
                old_phi = np.exp(-(np.linalg.norm(old_s - self.c[i - 12]) ** 2) / (2.0 * self.sigma ** 2))
            elif action == MountainCar.RIGHT and 24 <= i < 36:
                old_phi = np.exp(-(np.linalg.norm(old_s - self.c[i - 24]) ** 2) / (2.0 * self.sigma ** 2))
            else:
                old_phi = 0
         
            if new_action == MountainCar.LEFT and 0 <= i < 12:
                new_phi = np.exp(-(np.linalg.norm(new_s - self.c[i]) ** 2) / (2.0 * self.sigma ** 2))
            elif new_action == MountainCar.NEUTRAL and 12 <= i < 24:
                new_phi = np.exp(-(np.linalg.norm(new_s - self.c[i - 12]) ** 2) / (2.0 * self.sigma ** 2))
            elif new_action == MountainCar.RIGHT and 24 <= i < 36:
                new_phi = np.exp(-(np.linalg.norm(new_s - self.c[i - 24]) ** 2) / (2.0 * self.sigma ** 2))
            else:
                new_phi = 0
            
            self.X[self.X_num, i] = old_phi - self.gamma * new_phi
        
        self.reward[self.reward_num] = reward
        
        self.X_num += 1
        self.reward_num += 1

    def get_reward(self, x):
        return 1.0 / (1.0 + (0.5 - min(x, 0.5) ** 2))

    def step(self):
        action = self.decide_action(self.x, self.dx)
        self.x, self.dx = self.update_status(self.x, self.dx, action)
        reward = self.get_reward(self.x)
        
        self.update_matrix(self.x, self.dx, action, reward)
        
        #print "action: %d, x: %lf, dx: %lf, reward: %lf" % (action, x, dx, reward)
        if self.x > 0.5:
            print "***clear!!***"
            return True

    def process(self):
        for l in range(self.L):
            for m in range(self.M):
                self.init_status()
                if m % 10:
                    self.epsilon = 1.0
                else:
                    self.epsilon = 0.005
                for t in range(self.T):
                    if self.step():
                        self.plot()
                        ValuePlot(self.theta, self.sigma, self.c)
                        CarPlot(self.mem_x)
                    self.mem_x.append(self.x)
                print "%depoch\t\t%depisode\tmax x:%lf" % (l + 1, m + 1, max(self.mem_x))
            self.plot()
            self.theta = np.linalg.inv(self.X.transpose().dot(self.X) + np.eye(36) * 0.0000001).dot(self.X.transpose()).dot(self.reward)

    def plot(self):
        x = np.arange(-1.2, 0.6, 0.01)
        plt.plot(x, [np.sin(3.0 * _x) for _x in x], "r--")
        plt.plot(self.mem_x, [np.sin(3.0 * _x) for _x in self.mem_x], "bo")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    mountincar = MountainCar()
    mountincar.process()

