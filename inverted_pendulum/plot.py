import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class Penplot(object):
    def __init__(self, states, anime=False, fig=False):
        self.anime = anime
        self.fig = fig
        self.x = [state[0] for state in states]
        self.x_dot = [state[1] for state in states]
        self.theta = [state[2] for state in states]
        self.theta_dot = [state[3] for state in states]
        self.process()

    def plot(self, data):
        x, theta, frame = data
        self.time_text.set_text('time:%.2fs\nstep:%d' % (frame*0.02, frame))
        
        y = 0.05
        theta_x = x + math.sin(theta) * 0.25
        theta_y = y + math.cos(theta) * 0.25
        
        self.car.set_data(x, y/2.0)
        self.line.set_data((x, theta_x), (y, theta_y))

    def gen(self):
        for frame in xrange(len(self.x)):
            yield self.x[frame], self.theta[frame], frame

    def process(self):
        if self.anime:
            fig = plt.figure(figsize=(20, 4.5))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-0.1, 0.9)
            ax.grid()
            
            self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes) 
            self.car, = ax.plot([], [], 's', ms=15)
            self.line, = ax.plot([], [], 'b-', lw=2)
            
            ani = animation.FuncAnimation(fig, self.plot, self.gen, interval=1, repeat_delay=3000, repeat=True)
            
            try:
                plt.show()
            except:
                pass
        
        if self.fig:
            steps = xrange(len(self.x))
            
            plt.subplot(2, 1, 1)
            plt.title("x, theta")
            plt.plot(steps, self.x, label="x")
            plt.plot(steps, self.theta, label="theta")
            plt.legend(loc='upper left')
            
            plt.subplot(2, 1, 2)
            plt.title("x_dot, theta_dot")
            plt.plot(steps, self.x_dot, label="x_dot")
            plt.plot(steps, self.theta_dot, label="theta_dot")
            plt.legend(loc='upper left')
            
            try:
                plt.show()
            except:
                pass

