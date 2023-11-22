# Simone Mencarelli
# September 2023
# test the conformal array pattern creating a uniform rectangular array and comparing the pattern
# to a uniform aperture antenna with same polarization.

# %% includes
import numpy as np
import matplotlib.pyplot as plt
from conformal_array_pattern import ConformalArray
from radartools.farField import UniformAperture

# %% parameters
# antenna parameters
freq = 10e9
wavelength = 3e8 / freq
# elements spacing
dx = wavelength / 2
dy = wavelength / 2
# aperture size
L = 2
W = 0.3
# align aperture size to a multiple of dx dy
L = np.ceil(L / dx) * dx
W = np.ceil(W / dy) * dy
# print actual aperture size
print(f'Aperture size: {L} x {W} m')
# number of elements
Nx = int(L / dx)
Ny = int(W / dy)
# print actual number of elements
print(f'Number of elements: {Nx} x {Ny}')
# %% create uniform rectangular array
# 1. create a uniform aperture for array element
element = UniformAperture(dx, dy, freq)
# 2. location of elements in the array
x = np.arange(-np.ceil(Nx / 2), -np.ceil(Nx / 2) + Nx) * dx
y = np.arange(-np.ceil(Ny / 2), -np.ceil(Ny / 2) + Ny) * dy
x_mesh, y_mesh = np.meshgrid(x, y)
z_mesh = np.zeros_like(x_mesh)
# 3. radiation normal of the elements
norm_x = np.zeros_like(x_mesh)
norm_y = np.zeros_like(y_mesh)
norm_z = np.ones_like(x_mesh)
# 4. x-axis tangent to the element
tan_x = np.ones_like(x_mesh)
tan_y = np.zeros_like(y_mesh)
tan_z = np.zeros_like(x_mesh)
# 5. uniform excitation vector
excitation = np.ones_like(x_mesh)
# 6. create the array
array = ConformalArray(element,
                       x_mesh, y_mesh, z_mesh,
                       norm_x, norm_y, norm_z,
                       tan_x, tan_y, tan_z,
                       excitation, freq)

# %% create the equivalent uniform aperture
aperture = UniformAperture(dx * Nx, dy * Ny, freq)

# %% phi = 0 cut
theta = np.linspace(-np.pi / 2, np.pi / 2, 721)
phi = np.zeros_like(theta)
theta = theta.reshape((-1, 1))
phi = phi.reshape((-1, 1))
# compute the pattern for the array
array_e_t, array_e_p = array.far_field(theta, phi)
# normalise the pattern amplitude by the radiation intensity
array_e_t /= np.sqrt(array.radiated_power() * element.eta)
array_e_p /= np.sqrt(array.radiated_power() * element.eta)
# compute the pattern for the equivalent aperture
aperture_e_t, aperture_e_p = aperture.mesh_E_field_theor(theta, phi)
# normalise the pattern amplitude by the radiation intensity
aperture_e_t /= np.sqrt(aperture.get_radiated_power() * aperture.eta)
aperture_e_p /= np.sqrt(aperture.get_radiated_power() * aperture.eta)
# plot the patterns
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(array_e_t)), label='array')
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(aperture_e_t)), '--', label='aperture')
ax.set_xlabel(r'$\theta$ [deg]')
ax.set_ylabel(r'$E_{\theta}$ [dB]')
ax.grid()
ax.legend()
ax.set_title('phi = 0 cut')
fig.tight_layout()
# plot the patterns
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(array_e_p)), label='array')
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(aperture_e_p)), '--', label='aperture')
ax.set_xlabel(r'$\theta$ [deg]')
ax.set_ylabel(r'$E_{\phi}$ [dB]')
ax.grid()
ax.legend()
ax.set_title('phi = 0 cut')
fig.tight_layout()
plt.show()

# %% phi = 90 cut
theta = np.linspace(-np.pi / 2, np.pi / 2, 721)
phi = np.ones_like(theta) * np.pi / 2
theta = theta.reshape((-1, 1))
phi = phi.reshape((-1, 1))
# compute the pattern for the array
array_e_t, array_e_p = array.far_field(theta, phi)
# normalise the pattern amplitude by the radiation intensity
array_e_t /= np.sqrt(array.radiated_power() * element.eta)
array_e_p /= np.sqrt(array.radiated_power() * element.eta)
# compute the pattern for the equivalent aperture
aperture_e_t, aperture_e_p = aperture.mesh_E_field_theor(theta, phi)
# normalise the pattern amplitude by the radiation intensity
aperture_e_t /= np.sqrt(aperture.get_radiated_power() * aperture.eta)
aperture_e_p /= np.sqrt(aperture.get_radiated_power() * aperture.eta)
# plot the patterns
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(array_e_t)), label='array')
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(aperture_e_t)), '--', label='aperture')
ax.set_xlabel(r'$\theta$ [deg]')
ax.set_ylabel(r'$E_{\theta}$ [dB]')
ax.grid()
ax.legend()
ax.set_title('phi = 90 cut')
fig.tight_layout()
# plot the patterns
fig, ax = plt.subplots(1)
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(array_e_p)), label='array')
ax.plot(theta * 180 / np.pi, 20 * np.log10(np.abs(aperture_e_p)), '--', label='aperture')
ax.set_xlabel(r'$\theta$ [deg]')
ax.set_ylabel(r'$E_{\phi}$ [dB]')
ax.grid()
ax.legend()
ax.set_title('phi = 90 cut')
fig.tight_layout()
plt.show()

# %% 2-d pattern visualisation
theta = np.linspace(-np.pi / 8, np.pi / 8, 181)
phi = np.linspace(0, 2 * np.pi, 361)
T, P = np.meshgrid(theta, phi)
# compute the pattern for the array
array_e_t, array_e_p = array.far_field(T, P)
# normalise the pattern amplitude by the radiation intensity
array_e_t /= np.sqrt(array.radiated_power() * element.eta)
array_e_p /= np.sqrt(array.radiated_power() * element.eta)
# compute the pattern for the equivalent aperture
aperture_e_t, aperture_e_p = aperture.mesh_E_field_theor(T, P)
# normalise the pattern amplitude by the radiation intensity
aperture_e_t /= np.sqrt(aperture.get_radiated_power() * aperture.eta)
aperture_e_p /= np.sqrt(aperture.get_radiated_power() * aperture.eta)

# plot the patterns
# theta component array
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 20 * np.log10(np.abs(array_e_t)), cmap=plt.cm.plasma, vmin=-20, vmax=40,
                  rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('array E theta')
fig.tight_layout()
plt.show()
# phi component arrat
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 20 * np.log10(np.abs(array_e_p)), cmap=plt.cm.plasma, vmin=-20, vmax=40,
                  rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('array E phi')
fig.tight_layout()
plt.show()
# theta component aperture
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 20 * np.log10(np.abs(aperture_e_t)), cmap=plt.cm.plasma, vmin=-20,
                  vmax=40, rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('aperture E theta')
fig.tight_layout()
plt.show()
# phi component aperture
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 20 * np.log10(np.abs(aperture_e_p)), cmap=plt.cm.plasma, vmin=-20,
                  vmax=40, rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('aperture E phi')
fig.tight_layout()
plt.show()
# %% patterns difference
diff_theta = array_e_t - aperture_e_t
diff_phi = array_e_p - aperture_e_p
# plot the patterns
# theta component array
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 20 * np.log10(np.abs(diff_theta)), cmap=plt.cm.plasma, vmin=-1, vmax=1,
                  rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('diff E theta')
fig.tight_layout()
plt.show()
# phi component arrat
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 20 * np.log10(np.abs(diff_phi)), cmap=plt.cm.plasma, vmin=-1, vmax=1,
                  rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('diff E phi')
fig.tight_layout()
plt.show()

#%% gain plots
# compute the gain for the array
radiatedPower = array.radiated_power()
g_array = np.array(
    2 * np.pi * (np.abs(array_e_t) ** 2 + np.abs(array_e_p) ** 2) / (aperture.eta * radiatedPower),
    dtype=float)
# compute the gain for the aperture
g_aperture = np.array(
    2 * np.pi * (np.abs(aperture_e_t) ** 2 + np.abs(aperture_e_p) ** 2) / (aperture.eta * aperture.get_radiated_power()),
    dtype=float)
# plot the gains
# array
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 10 * np.log10(g_array), cmap=plt.cm.plasma, vmin=-10, vmax=40,
                  rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('array gain')
fig.tight_layout()
plt.show()
# aperture
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 10 * np.log10(g_aperture), cmap=plt.cm.plasma, vmin=-10, vmax=40,
                  rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('aperture gain')
fig.tight_layout()
plt.show()
# gain difference
diff_gain = g_array - g_aperture
# plot the gain difference
fig, ax = plt.subplots(1)
c = ax.pcolormesh(T * np.cos(P), T * np.sin(P), 10 * np.log10(diff_gain), cmap=plt.cm.plasma, vmin=-300, vmax=10,
                  rasterized=True)
ax.set_xlabel(r'$\theta\  cos \phi$')
ax.set_ylabel(r'$\theta\  sin \phi$')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
ax.set_title('gain difference')
fig.tight_layout()
plt.show()


