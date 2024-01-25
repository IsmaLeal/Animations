from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from scipy.integrate import solve_ivp

g = 9.8  # acceleration due to gravity (m/s^2)
t_stop = 25  # simulation end (s)
history_len = 30  # how many trajectory points to display
dt = 0.06
t = np.arange(0, t_stop, dt)

class Pendulum:
    def __init__(self, L1: float, L2: float, M1: float, M2: float, th1: float, th2: float, w1: float, w2: float):
        # Lengths (m)
        self.L = L1 + L2
        self.L1 = L1
        self.L2 = L2

        # Masses (kg)
        self.M1 = M1
        self.M2 = M2

        # Angles and angular speeds (degrees)
        self.th1 = th1
        self.th2 = th2
        self.w1 = w1
        self.w2 = w2

        self.state = np.radians([th1, w1, th2, w2])


    def derivs(self, t, state):
        '''
        Returns the system of ODEs describing a double pendulum
        '''
        self.dydx = np.zeros_like(state)

        self.dydx[0] = state[1]

        delta = state[2] - state[0]
        den1 = (self.M1+self.M2) * self.L1 - self.M2 * self.L1 * cos(delta) * cos(delta)
        self.dydx[1] = ((self.M2 * self.L1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + self.M2 * g * sin(state[2]) * cos(delta)
                    + self.M2 * self.L2 * state[3] * state[3] * sin(delta)
                    - (self.M1+self.M2) * g * sin(state[0]))
                   / den1)

        self.dydx[2] = state[3]

        den2 = (self.L2/self.L1) * den1
        self.dydx[3] = ((- self.M2 * self.L2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (self.M1+self.M2) * g * sin(state[0]) * cos(delta)
                    - (self.M1+self.M2) * self.L1 * state[1] * state[1] * sin(delta)
                    - (self.M1+self.M2) * g * sin(state[2]))
                   / den2)

        return self.dydx


    def positions(self):
        '''
        Returns positions of every mass blob
        '''
        self.y = solve_ivp(self.derivs, t[[0, -1]], self.state, t_eval=t).y.T

        self.x1 = self.L1 * sin(self.y[:, 0])
        self.y1 = -self.L1 * cos(self.y[:, 0])

        self.x2 = self.L2 * sin(self.y[:, 2]) + self.x1
        self.y2 = -self.L2 * cos(self.y[:, 2]) + self.y1

        return self.x1, self.y1, self.x2, self.y2


class Simulate:
    def __init__(self, *args):

        self.fig = plt.figure(figsize=(5, 4))
        self.ax = self.fig.add_subplot()
        self.ax.set_aspect('equal')

        self.time_template = 'time = %.1fs'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

        self.args = args

        self.lines = [None] * len(args)
        self.traces = [None] * len(args)
        self.historyx, self.historyy = [], []

        self.L = 1
        for idx, arg in enumerate(args):

            self.lines[idx], = self.ax.plot([], [], 'o-', lw=2)
            self.traces[idx], = self.ax.plot([], [], '-', lw=1)

            self.historyx.append(deque(maxlen=history_len))
            self.historyy.append(deque(maxlen=history_len))

            if arg.L > self.L:
                self.L = arg.L

        self.ax.set_xlim(-self.L, self.L)
        self.ax.set_ylim(-self.L, 1)


    def animate(self, i):
        for idx, line in enumerate(self.lines):
            x1, y1, x2, y2 = self.args[idx].positions()
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

            if i == 0:
                self.historyx[idx].clear()
                self.historyy[idx].clear()

            self.historyx[idx].appendleft(thisx[2])
            self.historyy[idx].appendleft(thisy[2])

            line.set_data(thisx, thisy)
            self.traces[idx].set_data(self.historyx[idx], self.historyy[idx])

        self.time_text.set_text(self.time_template % (i*dt))

        return self.lines, self.traces, self.time_text


p1 = Pendulum(1, 1, 1, 1, 90, 90.0, 0, 0)
p2 = Pendulum(1, 1, 1, 1, 90, 90.1, 0, 0)
p3 = Pendulum(1, 1, 1, 1, 90, 90.2, 0, 0)
p4 = Pendulum(1, 1, 1, 1, 90, 90.3, 0, 0)
p5 = Pendulum(1, 1, 1, 1, 90, 90.4, 0, 0)
p6 = Pendulum(1, 1, 1, 1, 90, 90.5, 0, 0)
p7 = Pendulum(1, 1, 1, 1, 90, 90.6, 0, 0)
p8 = Pendulum(1, 1, 1, 1, 90, 90.7, 0, 0)
p9 = Pendulum(1, 1, 1, 1, 90, 90.8, 0, 0)
p10 = Pendulum(1, 1, 1, 1, 90, 90.9, 0, 0)
p11 = Pendulum(1, 1, 1, 1, 90, 91, 0, 0)

sim = Simulate(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)
ani = animation.FuncAnimation(sim.fig, sim.animate, len(t), interval=1, blit=True)
ani.save('doublepends.gif', writer='pillow', fps=30)
plt.show()