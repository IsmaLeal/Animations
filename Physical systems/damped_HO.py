import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen():
    for cnt in itertools.count():
        t = cnt / 10
        yield t, np.sin(2*np.pi*t) * np.exp(-t/5)


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, c='red')
ax.grid()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
xdata, ydata = [], []


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    line.set_data(xdata, ydata)
    ax.set_title(f'Time: {round(t)}s')

    xmin, xmax = ax.get_xlim()
    if t >= xmax:
        ax.set_xlim(0, 1.5 * xmax)

    return line,


ani = animation.FuncAnimation(fig, run, data_gen, interval=15, init_func=init)
plt.show()