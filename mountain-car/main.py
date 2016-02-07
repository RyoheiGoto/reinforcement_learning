import numpy as np
import matplotlib.pyplot as plt
from plot import CarPlot

class MountainCar(object):
    def __init__(self):
        np.random.seed()
        
        self.tau = 0.1
        self.m = 0.2 #kg
        self.k = 0.3
        self.A = -0.2, 0, 0.2 #kgm/s^2	
        
        self.T = 50
        
        self.x = 0
        self.dx = 0
        
        self.c = np.array([ [-1.2, -1.5], [-1.2, -0.5], [-1.2, 0.5], [-1.2, 1.5],
                            [-0.35, -1.5], [-0.35, -0.5], [-0.35, 0.5], [-0.35, 1.5],
                            [0.5, -1.5], [0.5, -0.5], [0.5, 0.5], [0.5, 1.5 ]   ])
        
        self.theta = np.zeros(36)
        
        self.delta = 0.05
        self.gamma = 0.95
        self.epsilon = 0.005
        
        self.error = np.ndarray(shape=(20 * self.T, 36))
        self.e = 0
        self.reward = np.ndarray(20 * self.T)
        self.r = 0

    def init_status(self):
        self.x = -0.5
        self.dx = 0
        self.mem_x = [self.x]

    def decide_action(self, dx, x):
        s = np.array([x, dx])
        a0 = 0
        a1 = 0
        a2 = 0
        for i in range(12):
            a0 += self.theta[i] * np.exp(-(np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.delta ** 2))
            a1 += self.theta[i + 12] * np.exp(-(np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.delta ** 2))
            a2 += self.theta[i + 24] * np.exp(-(np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.delta ** 2))
        """
        if max([a0, a1, a2]) == a0 and a0:
            policy = [0.99, 0.005, 0.005]
        elif max([a0, a1, a2]) == a1 and a1:
            policy = [0.005, 0.99, 0.005]
        elif max([a0, a1, a2]) == a2 and a2:
            policy = [0.005, 0.005, 0.99]
        else:
            policy = [0.33, 0.33, 0.34]
        """
        a = a0, a1, a2
        if not sum(a):
            policy = [0.33, 0.33, 0.34]
        else:
            if max(a) == a0:
                policy = [1.0 - self.epsilon + self.epsilon / 3.0, self.epsilon / 3.0, self.epsilon / 3.0]
            elif max(a) == a1:
                policy = [self.epsilon / 3.0, 1.0 - self.epsilon + self.epsilon / 3.0, self.epsilon / 3.0]
            elif max(a) == a2:
                policy = [self.epsilon / 3.0, self.epsilon / 3.0, 1.0 - self.epsilon + self.epsilon / 3.0]
        
        tmp = 0
        for action, prob in enumerate(policy):
            tmp += prob
            if np.random.random() < tmp:
                return action
        else:
            return 2

    def update_status(self, dx, x, a):
        dx += (-9.8 * self.m * np.cos(3.0 * x) + self.A[a] / self.m - self.k * dx) * self.tau
        x += dx * self.tau
        
        return dx, x

    def update_error(self, dx, x, a, reward):
        newdx, newx = self.update_status(dx, x, a)
        s = np.array([newx, newdx])
        a0 = 0
        a1 = 0
        a2 = 0
        for j in range(12):
            a0 += self.theta[j] * np.exp(-(np.linalg.norm(s - self.c[j]) ** 2) / (2.0 * self.delta ** 2))
            a1 += self.theta[j + 12] * np.exp(-(np.linalg.norm(s - self.c[j]) ** 2) / (2.0 * self.delta ** 2))
            a2 += self.theta[j + 24] * np.exp(-(np.linalg.norm(s - self.c[j]) ** 2) / (2.0 * self.delta ** 2))
        
        if not a0 and not a1 and not a2:
            action = np.random.randint(0, 3)
        else:
            action = [a0, a1, a2].index(max([a0, a1, a2]))
        
        for i in range(36):
            s = np.array([x, dx])
            if a == 0 and  0 <= i < 12:
                old = np.exp(- (np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.delta ** 2))
            elif a == 1 and 12 <= i < 24:
                old = np.exp(- (np.linalg.norm(s - self.c[i - 12]) ** 2) / (2.0 * self.delta ** 2))
            elif a == 2 and 24 <= i < 36:
                old = np.exp(- (np.linalg.norm(s - self.c[i - 24]) ** 2) / (2.0 * self.delta ** 2))
            else:
                old = 0
         
            newdx, newx = self.update_status(dx, x, a)
            s = np.array([newx, newdx])
         
            if 0 <= i < 12 and action == 0:
                new = np.exp(- (np.linalg.norm(s - self.c[i]) ** 2) / (2.0 * self.delta ** 2))
            elif 12 <= i < 24 and action == 1:
                new = np.exp(- (np.linalg.norm(s - self.c[i - 12]) ** 2) / (2.0 * self.delta ** 2))
            elif 24 <= i < 36 and action == 2:
                new = np.exp(- (np.linalg.norm(s - self.c[i - 24]) ** 2) / (2.0 * self.delta ** 2))
            else:
                new = 0
     
            self.error[self.e, i] = old - self.gamma * new
        self.e += 1
        
        self.reward[self.r] = reward
        self.r += 1

    def get_reward(self, x):
        return 1.0 / (1.0 + (0.5 - x) ** 2)

    def step(self):
        action = self.decide_action(self.dx, self.x)
        dx, x = self.update_status(self.dx, self.x, action)
        reward = self.get_reward(x)
        
        self.update_error(dx, x, action, reward)
        
        self.dx = dx
        self.x = x
        
        #print "action: %d, x: %lf, dx: %lf, reward: %lf" % (action, x, dx, reward)
        if x > 0.5:
            print "clear!!",
            return True

    def process(self):
        for l in range(100):
            for m in range(20):
                print "%depoch, %depisode" % (l + 1, m + 1),
                self.init_status()
                for t in range(self.T):
                    if self.step():
                        mountcar.plot()
                        CarPlot(self.mem_x)
                        return
                    self.mem_x.append(self.x)
                print "max x:%lf" % max(self.mem_x)
            mountcar.plot()
            lsm = np.linalg.inv(self.error.transpose().dot(self.error) + np.eye(36) * 0.0000001).dot(self.error.transpose()).dot(self.reward)
            self.theta = lsm
            self.e = 0
            self.r = 0

    def plot(self):
        x = np.arange(-1.2, 0.6, 0.1)
        plt.plot(x, [np.sin(3.0 * _x) for _x in x], "r--")
        plt.plot(self.mem_x, [np.sin(3.0 * _x) for _x in self.mem_x], "bo")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    mountcar = MountainCar()
    mountcar.process()

