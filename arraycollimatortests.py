# %% Simone Mencarelli November 23

# %%
import numpy as np
from numpy import pi, sin, cos, tan
import scipy as sp
import matplotlib.pyplot as plt
from arrayfactor import AntennaArray

# %% Uniform linear array definition
lam = 3e8 / 10e9
dx = lam / 2
N = 23
x = np.arange(N) * dx - (N - 1) * dx / 2
exc = np.ones_like(x)
uniform_array = AntennaArray(x, np.zeros_like(x), np.zeros_like(x), exc, 10e9)

# %% Non-uniform linear array definition
# circular section angle
tc = 150 * pi / 180
# radius
r = x[-1] / (sin(tc / 2))
# individual points theta angle
t = np.linspace(-tc / 2, tc / 2, N)
# individual points x and y coordinates
xc = r * sin(t)
yc = r - r * cos(t)
non_uniform_array = AntennaArray(xc, np.zeros_like(xc), np.zeros_like(xc), exc, 10e9)

# %% Uniform curved array definition
curved_array = AntennaArray(xc, yc, np.zeros_like(xc), exc, 10e9)

# %% Plot of array element positions
fig, ax = plt.subplots(1)
ax.scatter(uniform_array.points[0], uniform_array.points[1], marker='^', label='Uniform linear array')
ax.scatter(non_uniform_array.points[0], non_uniform_array.points[1], marker='*', label='Non-uniform linear array')
ax.scatter(curved_array.points[0], curved_array.points[1], marker='x', label='Curved array')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_ylim(-0.1, 0.1)
ax.legend()
plt.show()

# %% Unitary excitation patterns comparison
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * 0
af_uniform = uniform_array.factor(theta, phi)
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N))
ax.legend()
plt.show()

# %% Plot of the array factors
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * pi/2
af_uniform = uniform_array.factor(theta, phi)
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array phi = 90*')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array phi = 90*')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array phi = 90*')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N)+1)
ax.legend()
plt.show()



# %% Non-Unitary excitation patterns comparison
# excitation generation algorithm 1
exc = np.zeros_like(x)
exc[0] = ((xc[1] - xc[0]) / 2)
for i in range(1, N - 1):
    exc[i] = ((xc[i + 1] - xc[i]) / 2 + (xc[i] - xc[i - 1]) / 2)
exc[-1] = ((xc[-1] - xc[-2]) / 2)
# conservation of power
sigma = np.sqrt(np.sum(exc**2) / N)
exc /= sigma
# Setting the excitations
non_uniform_array.excitations = exc
curved_array.excitations = exc
# Recomputing the array factors
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)

# %% Plot excitation patterns
fig, ax = plt.subplots(1)
ax.plot(xc, curved_array.excitations, label='Excitation curved array')
ax.plot(x, uniform_array.excitations, label='Excitation uniform array')
ax.set_xlabel('x [m]')
ax.set_ylabel('Excitation')
plt.show()

# %% Plot of the array factors
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * 0
af_uniform = uniform_array.factor(theta, phi)
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N))
ax.legend()
plt.show()

# %% Non-Unitary excitation patterns comparison
# excitation generation algorithm 2
exc = np.zeros_like(x)
for i in range(N):
    exc[i] = (sin(-tc / 2 + (i + 1 / 2) * tc / (N-1)) - sin(-tc / 2 + (i - 1 / 2) * tc / (N-1)))

# conservation of power
sigma = np.sqrt(np.sum(exc**2) / N)
exc /= sigma
# Setting the excitations
non_uniform_array.excitations = exc
curved_array.excitations = exc
# Recomputing the array factors
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
# %% Plot excitation patterns
fig, ax = plt.subplots(1)
ax.plot(xc, curved_array.excitations, label='Excitation curved array')
ax.plot(x, uniform_array.excitations, label='Excitation uniform array')
ax.set_xlabel('x [m]')
ax.set_ylabel('Excitation')
plt.show()

# %% Plot of the array factors
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * 0
af_uniform = uniform_array.factor(theta, phi)
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N))
ax.legend()
plt.show()

# %% Plot of the array factors
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * pi/2
af_uniform = uniform_array.factor(theta, phi)
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array phi = 90*')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array phi = 90*')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array phi = 90*')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N)+1)
ax.legend()
plt.show()

# %% Non-Unitary excitation patterns comparison
# excitation generation algorithm 3
exc = np.zeros_like(x)
for i in range(N):
    exc[i] = (sin(-tc / 2 + (i + 1 / 2) * tc / (N-1)) - sin(-tc / 2 + (i - 1 / 2) * tc / (N-1)))**(2)

# conservation of power
sigma = np.sqrt(np.sum(exc**2) / N)
exc /= sigma
# Setting the excitations
non_uniform_array.excitations = exc
curved_array.excitations = exc
# Recomputing the array factors
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
# %% Plot excitation patterns
fig, ax = plt.subplots(1)
ax.plot(xc, curved_array.excitations, label='Excitation curved array')
ax.plot(x, uniform_array.excitations, label='Excitation uniform array')
ax.set_xlabel('x [m]')
ax.set_ylabel('Excitation')
plt.show()

# %% Plot of the array factors
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * 0
af_uniform = uniform_array.factor(theta, phi)
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N))
ax.legend()
plt.show()

# %% Plot of the array factors
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * pi/2
af_uniform = uniform_array.factor(theta, phi)
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array phi = 90*')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array phi = 90*')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array phi = 90*')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N)+1)
ax.legend()
plt.show()

#%% excitation algorithm 4
# near field projection
# near field points = curved array points
excitations = uniform_array.near_field_projection(curved_array.points[0], curved_array.points[1]-r, curved_array.points[2])
excitations = np.abs(excitations)
# power conservation
sigma = np.sqrt(np.sum(excitations**2) / N)
excitations /= sigma
# set the excitations
curved_array.excitations = np.abs(excitations)
non_uniform_array.excitations = np.abs(excitations)

# %% Plot excitation patterns
fig, ax = plt.subplots(1)
ax.plot(xc, curved_array.excitations, label='Excitation curved array')
ax.plot(x, uniform_array.excitations, label='Excitation uniform array')
ax.set_xlabel('x [m]')
ax.set_ylabel('Excitation')
plt.show()
# %% Plot of the array factors
theta = np.linspace(-pi / 2, pi / 2, 360)
phi = np.ones_like(theta) * 0
# recompute the array factors
af_non_uniform = non_uniform_array.factor(theta, phi)
af_curved = curved_array.factor(theta, phi)
af_uniform = uniform_array.factor(theta, phi)
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_uniform)), label='Uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_non_uniform)), label='Non-uniform linear array')
ax.plot(theta * 180 / pi, 20 * np.log10(np.abs(af_curved)), '--', label='Curved array')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Array factor [dB]')
ax.set_ylim(-20, 20 * np.log10(N))
ax.legend()
plt.show()

