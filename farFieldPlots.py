# Some Nice plots of the far field data
# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from farFieldCST import *

matplotlib.use('Qt5Agg')

# %% User input
reference_pattern = 'patterns_good/undeformed.ffs'
distorted_pattern = 'patterns_good/deformed.ffs'

# %% load patterns
ant_ref = Aperture(reference_pattern)
ant_dist = Aperture(distorted_pattern)

# %% plot gain in main cuts copolar and crosspolar components
theta = np.linspace((-90) * np.pi / 180, (90) * np.pi / 180, 2501)
phi_E = np.array(0)
phi_H = np.array(np.pi / 2)
# E cut
print('e cut computing')
g_co_e, g_x_e = ant_ref.co_cross_polar_gain(theta, phi_E)
print('h cut computing')
g_co_h, g_x_h = ant_ref.co_cross_polar_gain(theta, phi_H)
# plot
fig, ax = plt.subplots(1)
cmap = plt.get_cmap("tab10").colors
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_co_e), '-', color=cmap[0], label='E-cut Co-pol nom.')
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_x_e), '--', color=cmap[0], label='E-cut X-pol nom.')
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_co_h), '-', color=cmap[1], label='H-cut Co-pol nom')
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_x_h), '--', color=cmap[1], label='H-cut X-pol nom')
print('e cut computing')
g_co_e, g_x_e = ant_dist.co_cross_polar_gain(theta, phi_E)
print('h cut computing')
g_co_h, g_x_h = ant_dist.co_cross_polar_gain(theta, phi_H)
# plot
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_co_e), '-', color=cmap[2], label='E-cut Co-pol def.')
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_x_e), '--', color=cmap[2], label='E-cut X-pol def.')
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_co_h), '-', color=cmap[3], label='H-cut Co-pol def.')
ax.plot(theta * 180 / np.pi, 10 * np.log10(g_x_h), '--', color=cmap[3], label='H-cut X-pol def.')

ax.set_xlabel('$\Theta$ [deg]')
ax.set_ylabel('Gain ')
ax.set_title('Gain in main cuts')
ax.legend(ncol=2)
plt.show()

# %% now for the 2-d plots

# reference pattern
theta = np.linspace(0, 20 * np.pi / 180, 1001)
phi = np.linspace(0, 2 * np.pi, 571)
T, P = np.meshgrid(theta, phi)
g_co_r, g_x_r = ant_ref.co_cross_polar_gain(T, P)
g_co_d, g_x_d = ant_dist.co_cross_polar_gain(T, P)
#%%
plt.style.use('dark_background')
fig, ax = plt.subplots(1)
plt.style.use('dark_background')
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 10*np.log10(g_co_r), cmap=plt.cm.plasma, vmin=-40, vmax=40, rasterized=True)
#ax.pcolormesh(T * np.cos(P), T * np.sin(P), (gain_r))
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_aspect('equal', 'box')
fig.savefig("niceplotsforpaper/ref_pattern_co.svg", format="svg", dpi=600)
plt.show()


fig, ax = plt.subplots(1)
plt.style.use('dark_background')
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 10*np.log10(g_x_r), cmap=plt.cm.plasma, vmin=-40, vmax=40, rasterized=True)
#ax.pcolormesh(T * np.cos(P), T * np.sin(P), (gain_r))
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_aspect('equal', 'box')
fig.savefig("niceplotsforpaper/ref_pattern_cross.svg", format="svg", dpi=600)
plt.show()
#%%
fig, ax = plt.subplots(1)
plt.style.use('dark_background')
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 10*np.log10(g_co_d), cmap=plt.cm.plasma, vmin=-40, vmax=40, rasterized=True)
#ax.pcolormesh(T * np.cos(P), T * np.sin(P), (gain_r))
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_aspect('equal', 'box')
fig.savefig("niceplotsforpaper/dist_pattern_co.svg", format="svg", dpi=600)
plt.show()


fig, ax = plt.subplots(1)
plt.style.use('dark_background')
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 10*np.log10(g_x_d), cmap=plt.cm.plasma, vmin=-40, vmax=40, rasterized=True)
#ax.pcolormesh(T * np.cos(P), T * np.sin(P), (gain_r))
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_aspect('equal', 'box')
fig.savefig("niceplotsforpaper/dist_pattern_cross.svg", format="svg", dpi=600)
plt.show()