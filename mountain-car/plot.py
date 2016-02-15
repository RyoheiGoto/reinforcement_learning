import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class CarPlot(object):
    def __init__(self, x):
        self.x = x
        self._process()

    def _plot(self, data):
        x, frame = data
        self.time_text.set_text('time:%.2fs\nstep:%d' % (frame*0.1, frame))
        
        self.car.set_data(x, np.sin(3.0 * x))

    def _gen(self):
        for frame in range(len(self.x)):
            yield self.x[frame], frame

    def _process(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid()
        
        x = np.arange(-1.2, 0.6, 0.01)
        plt.plot(x, [np.sin(3.0 * _x) for _x in x], "r--")
        
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes) 
        self.car, = ax.plot([], [], 'bo', ms=10)
        
        ani = animation.FuncAnimation(fig, self._plot, self._gen, interval=10, repeat_delay=3000, repeat=True)
        
        plt.show()

class ValuePlot(object):
    def __init__(self, theta, sigma, c):
        self.th = theta
        self.sigma = sigma
        self.c = c
        
        self._process()

    def _process(self):
        x = np.arange(-1.2, 0.5, 0.1)
        dx = np.arange(-1.5, 1.5, 0.1)
        X, DX = np.meshgrid(x, dx)
        
        Z0 = np.zeros_like(X)
        Z1 = np.zeros_like(X)
        Z2 = np.zeros_like(X)
       
        for j, _dx in enumerate(dx):
            for i, _x in enumerate(x):
                s = np.array([_x, _dx])
                q0 = 0.001 * sum([self.th[k] * np.exp(-(np.linalg.norm(s - self.c[k]) ** 2) / (2.0 * self.sigma ** 2)) for k in range(12)])
                q1 = 0.001 * sum([self.th[k + 12] * np.exp(-(np.linalg.norm(s - self.c[k]) ** 2) / (2.0 * self.sigma ** 2)) for k in range(12)])
                q2 = 0.001 * sum([self.th[k + 24] * np.exp(-(np.linalg.norm(s - self.c[k]) ** 2) / (2.0 * self.sigma ** 2)) for k in range(12)])
                tmp = np.exp(q0) + np.exp(q1) + np.exp(q2)
                p0 = np.exp(q0) / tmp
                p1 = np.exp(q1) / tmp
                p2 = np.exp(q2) / tmp
                policy = p0, p1, p2
                if max(policy) is p0:
                    Z0[j][i] = 1
                    Z1[j][i] = 0
                    Z2[j][i] = 0
                if max(policy) is p1:
                    Z0[j][i] = 0
                    Z1[j][i] = 1
                    Z2[j][i] = 0
                if max(policy) is p2:
                    Z0[j][i] = 0
                    Z1[j][i] = 0
                    Z2[j][i] = 1
        
        fig = plt.figure()
        
        ax = fig.add_subplot(221, projection='3d')
        ax.plot_wireframe(X, DX, Z0, color="blue")
        ax.set_xlabel("x[m]")
        ax.set_ylabel("dx[m/sec]")
        plt.title(r"left")
        
        bx = fig.add_subplot(222, projection='3d')
        bx.plot_wireframe(X, DX, Z1, color="red")
        bx.set_xlabel("x[m]")
        bx.set_ylabel("dx[m/sec]")
        plt.title(r"neutral")
        
        cx = fig.add_subplot(223, projection='3d')
        cx.plot_wireframe(X, DX, Z2, color="green")
        cx.set_xlabel("x[m]")
        cx.set_ylabel("dx[m/sec]")
        plt.title(r"right")
        
        ex = fig.add_subplot(224, projection='3d')
        ex.plot_wireframe(X, DX, Z0, color="blue")
        ex.plot_wireframe(X, DX, Z1, color="red")
        ex.plot_wireframe(X, DX, Z2, color="green")
        ex.set_xlabel("x[m]")
        ex.set_ylabel("dx[m/sec]")
        
        plt.show()

