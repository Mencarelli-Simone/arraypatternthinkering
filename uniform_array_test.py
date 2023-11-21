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

#%% phi = 90 cut
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
