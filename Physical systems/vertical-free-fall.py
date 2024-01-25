import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
g = 9.81                                        # Acceleration due to gravity (m/s^2)
fast_v = 10                           # Initial velocity (m/s)
slow_v = 5

# Time values
t_max = 1.45                                    # End of animation (s)
t_vals = np.linspace(0, t_max, 80)

# x-position values for each object
x_static = np.zeros(len(t_vals))
x_fast = fast_v * t_vals
x_slow = slow_v * t_vals

# Calculate vertical positions (y) for each object
y = -0.5 * g * t_vals ** 2


# Create the figure and axes for the plots
fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
ax1.set_xlim(-1, 15)     # for velocities under 10 m/s
ax1.set_ylim(-10, 0)
ax1.set_xlabel('Horizontal Position (m)')
ax1.set_ylabel('Vertical Position (m)')
ax1.set_title('Free fall for different horizontal velocities')

# Plot the objects (lines with a dot representing the falling objects)
static_line, = ax1.plot([], [], 'b', marker='o', label=r'Static: $v_x = 0$ m/s')
fast_line, = ax1.plot([], [], 'r', marker='o', label=r'Slow: $v_x = 5$ m/s')
slow_line, = ax1.plot([], [], 'g', marker='o', label=r'Fast: $v_x = 10$ m/s')


# Function to update the animation. Needs to return line objects (ax.plot unpacked)
def update(frame):
    static_line.set_data(x_static[frame], y[frame])
    fast_line.set_data(x_fast[frame], y[frame])
    slow_line.set_data(x_slow[frame], y[frame])
    return static_line, fast_line, slow_line


# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_vals), interval=5)

# Save it
ani.save('a.gif', writer='pillow', fps=30)

plt.tight_layout()
plt.legend()
plt.show()
