# Simone Mencarelli
# November 2023
# Test the conformal array pattern creating a cylindrical conformal array.
# compare to a uniform aperture antenna with same aperture area.
# it is assumed that the array has elements equally spaced along the circumference

# %% includes
import numpy as np
from numpy import pi, sin, cos, tan, arcsin
import matplotlib.pyplot as plt
from conformal_array_pattern import ConformalArray
from radartools.farField import UniformAperture
import matplotlib

# set pyqt5 render
matplotlib.use('Qt5Agg')
# %% parameters
#  CYLINDRICAL aperture definition
# antenna parameters
freq = 10e9
wavelength = 3e8 / freq
# elements spacing
dx = wavelength / 2
dc = wavelength / 2  # spacing on conformal surface
# aperture size
L = 0.6
W = 0.3
# circular section angle
tc = 150 * pi / 180

## derivates
# radius
r = W / (sin(tc / 2) * 2)
# For the length is easy to define the number of elements
# align aperture size to a multiple of dx dy
L = np.ceil(L / dx) * dx
# number of elements in length
Nx = int(L / dx)
# for the width we first find the closest approximation of the number of elements
dt = 2 * arcsin(dc / (2 * r))
# the number of radial segments is then (defect approximation)
Nt = int(tc / dt)
# make it odd to have a central element
if Nt % 2 == 0:
    Nt += 1
# actual circular section
tc = Nt * dt
# actual width
W = 2 * r * sin(tc / 2)
# print actual length and width limiting the number of digits to two after the point
print(f'Aperture size: {L:.4f} x {W:.4f} m')
# print actual number of elements
print(f'Number of elements: {Nx} x {Nt}')
# print actual subtended angle and radius
print(f'Subtended angle: {tc * 180 / pi :.4f} deg')
print(f'Radius: {r:.4f} m')

# %% Array geometry
# 1. location of elements in the array, separating the problem in length and section
# length
xc = np.arange(-int(Nx / 2), -int(Nx / 2) + Nx) * dx
# theta
t = np.arange(-int(Nt / 2), -int(Nt / 2) + Nt) * dt
# xc repeats along the theta axis
xc_mesh, t_mesh = np.meshgrid(xc, t)
# individual points y and z coordinates
yc_mesh = r * sin(t_mesh)
zc_mesh = r - r * cos(t_mesh)
# 2. radiation normal of the elements in the circular section
norm_x = np.zeros_like(t_mesh)
# the norm lies in the y-z plane
norm_y = -sin(t_mesh)
norm_z = cos(t_mesh)
# 3. x-axis tangent to the element, i.e. global x-axis
tan_x = np.ones_like(t_mesh)
tan_y = np.zeros_like(t_mesh)
tan_z = np.zeros_like(t_mesh)

# %% create the array
# uniform excitation vector
excitation = np.ones_like(t_mesh)
# create half wavelength elemetal aperture
element = UniformAperture(dx, dx, freq)
# create the array
array = ConformalArray(element,
                       xc_mesh, yc_mesh, zc_mesh,
                       norm_x, norm_y, norm_z,
                       tan_x, tan_y, tan_z,
                       excitation, freq)
# plot the geometry in 3d using the embedded functions
# create 3d fig and axis
fig, ax = plt.subplots(1, subplot_kw={'projection': '3d'})
# plot points on ax
array.plot_points(ax)
# plot the lcs on ax
array.plot_lcs(ax, length=dx / 2)
plt.show()
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(0, 2)
