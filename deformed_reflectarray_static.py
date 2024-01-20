# this file creates a reflectarray from an abaqus mesh

# %% import
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as ml
from reflectarray_model import ReflectArray, RACell
from conformal_array_pattern import ConformalArray
from feed_antenna_pattern import FeedAntenna
from AbaqusMeshGridReader import AbaqusMesh
from radartools.farField import UniformAperture

# %% user input
# abaqus mesh file
filename = 'random_analysis_results/res_1mode.txt'
rows = 3312
columns = 11
xpoints = 23
ypoints = 144
# y points valid indexes
y_idx = np.linspace(0, ypoints - 12, ypoints - 11, dtype=int)
## for the array
# cell size
dx = 0.015
# feed distance from surface (center feed)
rf = 1  # m
# frequency
freq = 10e9  # Hz

# %% unit cell element
element = UniformAperture(dx, dx, freq)

# %% array lcs and points extraction
# create an abaqus mesh object
mesh = AbaqusMesh(filename, rows, columns, xpoints, ypoints)
mesh.half_symmetry()  # to use for the current file, todo not use it
# extract the origins
xc, yc, zc, xc_d, yc_d, zc_d = mesh.get_cell_centers()
# extract the lcss
x1, x2, x3, y1, y2, y3, z1, z2, z3, x1_def, x2_def, x3_def, y1_def, y2_def, y3_def, z1_def, z2_def, z3_def = \
    mesh.get_cell_lcs()

# %% coordinates rearrangement and points centering, creation of the array objects
# y becomes z, z becomes y and x is the same
# 1. undeformed
# center positions
xc = xc[y_idx, :]
yc1 = zc[y_idx, :]
zc = yc[y_idx, :]
yc = yc1[y_idx, :]
# centering of the array on x axis
xc = xc - np.mean(xc)
# lcs tangential vector x component
tan1 = x1[y_idx, :]
tan2 = x3[y_idx, :]
tan3 = x2[y_idx, :]
# lcs normal vector z component
norm1 = z1[y_idx, :]
norm2 = z3[y_idx, :]
norm3 = z2[y_idx, :]
# creation of the undeformed array
array_undef = ConformalArray(element,
                             xc, yc, zc,
                             norm1, norm2, norm3,
                             tan1, tan2, tan3,
                             np.ones_like(xc), freq)
# 2. deformed
# center positions
xc_d = xc_d[y_idx, :]
yc1_d = zc_d[y_idx, :]
zc_d = yc_d[y_idx, :]
yc_d = yc1_d[y_idx, :]
# centering of the array on x axis
xc_d = xc_d - np.mean(xc_d)
# lcs tangential vector x component
tan1_d = x1_def[y_idx, :]
tan2_d = x3_def[y_idx, :]
tan3_d = x2_def[y_idx, :]
# lcs normal vector z component
norm1_d = z1_def[y_idx, :]
norm2_d = z3_def[y_idx, :]
norm3_d = z2_def[y_idx, :]
# creation of the deformed array
array_def = ConformalArray(element,
                           xc_d, yc_d, zc_d,
                           norm1_d, norm2_d, norm3_d,
                           tan1_d, tan2_d, tan3_d,
                           np.ones_like(xc_d), freq)
# %% feed antenna (same for both)
# feed antenna position
feed = FeedAntenna(0, 0, rf, 0, 0, -1, 1, 0, 0, freq, pol='x')

# %% reflectarrays creation
# element cell (same for both)
cell = RACell()
# undeformed reflectarray
ref_undef = ReflectArray(cell, feed, array_undef)
# deformed reflectarray
ref_def = ReflectArray(cell, feed, array_def)

# %% geometric plotting
# set elements wireframe
# elemental wireframe
wf_x = np.array([-dx / 2, dx / 2, dx / 2, -dx / 2, -dx / 2])
wf_y = np.array([-dx / 2, -dx / 2, dx / 2, dx / 2, -dx / 2])
wf_z = np.array([0, 0, 0, 0, 0])
ref_def.array.set_element_wireframe(wf_x, wf_y, wf_z)
ref_undef.array.set_element_wireframe(wf_x, wf_y, wf_z)
# element surface
# create the surface
x = np.array([-dx / 2, dx / 2])
y = np.array([-dx / 2, dx / 2])
z = np.array([[0, 0], [0, 0]])
ref_def.array.set_element_surface(x, y, z)
ref_undef.array.set_element_surface(x, y, z)
# undeformed
ml.figure(1, bgcolor=(0, 0, 0))
# clear
ml.clf()
# meshgrid
ref_undef.draw_reflectarray()
# lcs
ref_undef.array.plot_lcs_mayavi(length=dx / 2)
# feed lcs
ref_undef.feed.plot_lcs(scale_factor=dx / 2)
# deformed
ml.figure(2, bgcolor=(0, 0, 0))
# clear
ml.clf()
ref_def.draw_reflectarray()
ref_undef.array.draw_elements_mayavi(color=(.2, .2, .2))
# lcs
ref_def.array.plot_lcs_mayavi(length=dx / 2)
ml.show()

# %% collimate the reflectarray in the nominal configuration
phase_shifts = ref_undef.collimate_beam(0, 0)
# set the same phase shifts to the deformed reflectarray
ref_def.phase_shift = phase_shifts
# compute the status vector
ref_def.__update__(collimate=False, reflect=True)
ref_undef.__update__(collimate=False, reflect=True)
# %% far field calculation
from numpy import pi

theta = np.linspace(-pi / 8, pi / 8, 1800)
phi1 = np.ones_like(theta) * pi / 2
phi0 = np.zeros_like(theta)
# undeformed
Gco0, Gcross0 = ref_undef.directive_gain(theta, phi0, polarization='x')
Gco1, Gcross1 = ref_undef.directive_gain(theta, phi1, polarization='x')
# deformed
Gco0_d, Gcross0_d = ref_def.directive_gain(theta, phi0, polarization='x')
Gco1_d, Gcross1_d = ref_def.directive_gain(theta, phi1, polarization='x')
# %% plotting
plt.figure(3)
plt.clf()
plt.plot(theta * 180 / pi, 10*np.log10(Gco0), '--', label='co-pol')
plt.plot(theta * 180 / pi, 10*np.log10(Gcross0), '--', label='cross-pol')
plt.plot(theta * 180 / pi, 10*np.log10(Gco1), '--', label='co-pol')
plt.plot(theta * 180 / pi, 10*np.log10(Gcross1), '--', label='cross-pol')
# deformed
plt.plot(theta * 180 / pi, 10*np.log10(Gco0_d), label='co-pol')
plt.plot(theta * 180 / pi, 10*np.log10(Gcross0_d), label='cross-pol')
plt.plot(theta * 180 / pi, 10*np.log10(Gco1_d), label='co-pol')
plt.plot(theta * 180 / pi, 10*np.log10(Gcross1_d), label='cross-pol')
plt.legend()
plt.xlabel('theta (deg)')
plt.ylabel('Gain (dB)')
plt.show()
