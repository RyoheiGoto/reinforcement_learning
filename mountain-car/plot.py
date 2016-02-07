import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

        x = np.arange(-1.2, 0.6, 0.1)
        plt.plot(x, [np.sin(3.0 * _x) for _x in x], "r--")
        
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes) 
        self.car, = ax.plot([], [], 'bo', ms=10)
        
        ani = animation.FuncAnimation(fig, self._plot, self._gen, interval=10, repeat_delay=3000, repeat=True)

        try:
            plt.show()
        except AttributeError:
            pass

