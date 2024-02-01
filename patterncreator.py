# this file is used to create an antenna pattern starting from a shape file and reflectaray geometry
# the antenna pattern is then saved in a ffs file

# %% import
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import mayavi.mlab as ml
from reflectarray_model import ReflectArray, RACell
from conformal_array_pattern import ConformalArray
from feed_antenna_pattern import FeedAntenna
from abaqus_mesh_grid_reader import AbaqusMesh
from radartools.farField import UniformAperture
from ffsFileWriter import ffsWrite
from tqdm import tqdm

# %% user input
# abaqus mesh file
filename = 'random_analysis_results/res_1mode.txt'
rows = 3312
columns = 11
xpoints = 23
ypoints = 144
# y points valid indexes
y_idx = np.linspace(0, ypoints - 12, ypoints - 11, dtype=int)

## output files
# undeformed
undeformed_file = 'undeformeddddddd.ffs'
# deformed
deformed_file = 'deformedddddddddd.ffs'

## for the array
# cell size
dx = 0.015
# feed distance from surface (center feed)
rf = 1  # m

# frequency
freq = 10e9  # Hz

# Undeformed on or off
undeformed = True  # if false, only the deformed shape is computed, if true both are exported

# Surface plots on or off
surface_plots = False
if ~surface_plots:
    print("surface plots are off,no phase visualization or power visualization")
else:
    print("surface plots are on, plotting might be very slow")

## create a theta phi meshgrid ( just the positive hemisphere is fine)
theta = np.linspace(0, np.pi / 2, 1801)
phi = np.linspace(0, 2 * np.pi, 1801)
T, P = np.meshgrid(theta, phi)

# %% unit cell element
element = UniformAperture(dx, dx, freq)

# %% array lcs and points extraction
# create an abaqus mesh object
mesh = AbaqusMesh(filename, rows, columns, xpoints, ypoints)
mesh.half_symmetry()  # to use for the current file, todo not use it if different from mode 1
# mesh.scale(5,5,5)
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
# # undeformed
# ml.figure(1, bgcolor=(60/256, 60/256, 60/256))
# # clear
# ml.clf()
# # meshgrid
# ref_undef.draw_reflectarray()
# # lcs
# ref_undef.array.plot_lcs_mayavi(length=dx / 2)
# # feed lcs
# ref_undef.feed.plot_lcs(scale_factor=dx / 2)
# # deformed
ml.figure(2, bgcolor=(60 / 256, 60 / 256, 60 / 256))
# clear
ml.clf()
ref_def.draw_reflectarray(color=(0.1, 0.1, 0.8))
ref_undef.feed.plot_lcs(scale_factor=dx * 2, mode='arrow')
ref_undef.array.draw_elements_mayavi(color=(1, 1, 1))
# lcs
ref_def.array.plot_lcs_mayavi(length=dx / 2, mode='arrow')
ml.show()

# %% collimate the reflectarray in the nominal configuration
print("collimating the reflectarray ...")
phase_shifts = ref_undef.collimate_beam(0, 0)
# set the same phase shifts to the deformed reflectarray
ref_def.phase_shift = phase_shifts
# compute the status vector
ref_def.__update__(collimate=False, reflect=True)
ref_undef.__update__(collimate=False, reflect=True)
# % %debugg plotting, check the phase shift in the two reflectarrays
if surface_plots:
    # create mayavi figure for the undeformed
    ml.figure(4, bgcolor=(0, 0, 0))
    # clear
    ml.clf()
    # draw the reflectarray undeformed
    ref_undef.draw_reflectarray()
    # draw the surfaces
    if surface_plots:
        print("drawing surfaces ...")
        ref_undef.array.draw_element_surfaces_mayavi(parameter=ref_undef.phase_shift / (2 * pi))
    # create mayavi figure for the deformed
    ml.figure(5, bgcolor=(0, 0, 0))
    # clear
    ml.clf()
    # draw the reflectarray deformed
    ref_def.draw_reflectarray()
    # draw the surfaces
    if surface_plots:
        print("drawing surfaces ...")
        ref_def.array.draw_element_surfaces_mayavi(parameter=ref_def.phase_shift / (2 * pi))
    # show
    ml.show()

# %% compute patterns
# update
ref_undef.__update__(collimate=False, reflect=True)
ref_def.__update__(collimate=False, reflect=True)
if undeformed:
    # compute the power
    P_undef = np.sum(ref_undef.radiated_power_per_element())
    print('undeformed pattern calculation')
    # todo segment the calculation in blocks to avoid memory errors
    # allocate mem
    Et_undef = np.zeros_like(T, dtype=complex)
    Ep_undef = np.zeros_like(T, dtype=complex)
    # line by line segmentation
    for i in tqdm(range(len(phi))):
        Et_undef[i, :], Ep_undef[i, :] = ref_undef.far_field(T[i, :], P[i, :])
    ffsWrite(T.reshape(-1) * 180 / np.pi, P.reshape(-1) * 180 / np.pi,
             Et_undef.reshape(-1), Ep_undef.reshape(-1), len(phi), len(theta),
             undeformed_file,
             radiated_power=P_undef,
             accepted_power=P_undef,
             stimulated_power=P_undef,
             frequency=1e10)
print('deformed pattern calculation')
# todo segment the calculation in blocks to avoid memory errors
# allocate mem
Et_def = np.zeros_like(T, dtype=complex)
Ep_def = np.zeros_like(T, dtype=complex)
# line by line segmentation
for i in tqdm(range(len(phi))):
    Et_def[i, :], Ep_def[i, :] = ref_def.far_field(T[i, :], P[i, :])
P_def = np.sum(ref_def.radiated_power_per_element())
# save
ffsWrite(T.reshape(-1) * 180 / np.pi, P.reshape(-1) * 180 / np.pi,
         Et_def.reshape(-1), Ep_def.reshape(-1), len(phi), len(theta),
         deformed_file,
         radiated_power=P_def,
         accepted_power=P_def,
         stimulated_power=P_def,
         frequency=1e10)

# %%
