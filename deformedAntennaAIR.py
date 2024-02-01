# Simone Mencarelli
# September 2023
# Azimuth impulse response for deformed antenna
import matplotlib.pyplot as plt

# %% includes
from radartools.spherical_earth_geometry_radar import *
from radartools.design_functions import *
from farFieldCST import Aperture
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift

# %% User input
# broadside incidence angle
incidence_broadside = 25 * np.pi / 180
squint = 0
altitude = 500e3
# frequency
f = 10e9
# antenna length (or equivalent for 3db beamwidth)
La = 2
# speed of light
c = 299792458.0
# range swath
swath = 100e3
# pattern files
# reference_pattern = 'dummyReference.ffs'
# distorted_pattern = 'dummyDistortedMode2.ffs'
# reference_pattern = 'lyceanem/NormalAntennav2.ffe'
# distorted_pattern = 'lyceanem/DeformedAntennav2.ffe'
reference_pattern = 'patterns_good/undeformed.ffs'
distorted_pattern = 'patterns_good/deformed.ffs'

# %% set scene geometry
radarGeo = RadarGeometry()
# set rotation
looking_angle = incidence_angle_to_looking_angle(incidence_broadside, altitude)
radarGeo.set_rotation(looking_angle, 0, squint)
# set position
radarGeo.set_initial_position(0, 0, altitude)
# set orbital speed
v_s = radarGeo.orbital_speed()
radarGeo.set_speed(v_s)
# that's it
# %% load antenna patterns
# %% load patterns
ant_ref = Aperture(reference_pattern)
ant_dist = Aperture(distorted_pattern)

# %% calculation
# 1 Doppler Bandwidth (nominal 3dB beamwidth)
Bd = nominal_doppler_bandwidth(La, incidence_broadside, c / f, v_s, altitude)
print('Doppler Bandwidth: ', Bd)
# 2 Doppler Axis Incidence Axis
doppler = np.linspace(-Bd * 400, Bd * 400, 1000001)
# incidence at swath edges
r0, rg0 = range_from_theta(incidence_broadside * 180 / np.pi, altitude)
rgNF = np.array((rg0 - swath / 2, rg0 + swath / 2))
rNF = range_ground_to_slant(rgNF, altitude)
rgNF, incNF = range_slant_to_ground(rNF, altitude)
# incidence axis
incidence = np.linspace(incNF[0], incNF[1], 101)  # random length
# 3 Incidence Doppler meshgrid
I, D = np.meshgrid(incidence, doppler)
# 4 Incidence Azimuth Stationary time
I, A, Tk = mesh_doppler_to_azimuth(I, D, c / f, v_s, altitude)
# 5 GCS
X, Y, Z = mesh_incidence_azimuth_to_gcs(I, A, c / f, v_s, altitude)
# 6 LCS
Xl, Yl, Zl = mesh_gcs_to_lcs(X, Y, Z, radarGeo.Bc2s, radarGeo.S_0)
# 7 LCS spherical
R, T, P = meshCart2sph(Xl, Yl, Zl)
# 8 Antenna patterns
T[np.isnan(T)] = 0
P[np.isnan(P)] = 0  # to be safe
# Isolating co-polar component
# G_ref = ant_ref.mesh_gain_pattern(T, P) / ant_ref.max_gain()
G_ref, G_refx = ant_ref.co_cross_polar_gain(T, P)
G_ref /= np.max(G_ref)
# G_dist = ant_dist.mesh_gain_pattern(T, P) / ant_ref.max_gain()
G_dist , G_distx = ant_dist.co_cross_polar_gain(T, P)
G_dist /= np.max(G_ref)

# 9 AIR
# ifft in azimuth
air = np.where(np.abs(D) < Bd / 2, G_dist / G_ref, 0)
air[np.isnan(air)] = 0
airr = np.where(np.abs(D) < Bd / 2, 1, 0)  # ideal impulse response (reference case)
AIR = (doppler[-1] - doppler[0]) / (Bd) * ifftshift(ifft(ifftshift(air), axis=0))
AIRR = (doppler[-1] - doppler[0]) / (Bd) * ifftshift(ifft(ifftshift(airr), axis=0))

# %% 10 time to azimuth
# time axis
if len(doppler) % 2 == 0:
    t = np.linspace(-len(doppler) / (2 * (doppler[-1] - doppler[0])), len(doppler) / (2 * (doppler[-1] - doppler[0])),
                    len(doppler))
else:
    t = np.linspace(-(len(doppler) - 1) / (2 * (doppler[-1] - doppler[0])),
                    (len(doppler) + 1) / (2 * (doppler[-1] - doppler[0])),
                    len(doppler))
I, Ts = np.meshgrid(incidence, t)
# time to azimuth (it is not the same A of above)
I, A = mesh_incidence_time_to_incidence_azimuth(I, Ts, v_s, altitude)

# %% plot settings
fw = 5
fh = fw / np.sqrt(2)
fontsize = 7

# %% plot antenna pattern on Incidence Doppler (rcmc'ed)
plt.style.use('dark_background')
fig, ax = plt.subplots(1)
# puncture the indexes (one sample every 8)
jj = np.arange(0, len(I[:, 0]), 8).astype('int')
jj = np.argwhere(np.abs(D[:, 50]) < Bd * 3)[:, 0]
c = ax.pcolormesh(I[jj, :] * 180 / np.pi, D[jj, :] * 1e-3,
                  20 * np.log10(np.abs(G_dist[jj, :]) / np.max(np.abs(G_dist[jj, :]))), vmin=-50, vmax=20, rasterized=True)
ax.set_ylabel('Doppler shift [kHz]')
ax.set_xlabel('incidence angle [deg]')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
# Bd lines
ax.plot(np.array((incidence[0], incidence[-1])) * 180 / np.pi, np.ones(2) * Bd / 2e3, 'r')
ax.plot(np.array((incidence[0], incidence[-1])) * 180 / np.pi, -np.ones(2) * Bd / 2e3, 'r', label='Bd')
ax.legend()
fig.set_figwidth(fw)
fig.set_figheight(fh)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)
for t in (cbar.ax.get_yticklabels() + [cbar.ax.yaxis.label]):
    t.set_fontsize(fontsize)
plt.tight_layout()
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig("niceplotsforpaper/dist_on_doppler.svg", format="svg", dpi=600)
plt.show()

# %% plot antenna pattern on Incidence Doppler (rcmc'ed)
fig, ax = plt.subplots(1)
# puncture the indexes (one sample every 8)
jj = np.arange(0, len(I[:, 0]), 8).astype('int')
jj = np.argwhere(np.abs(D[:, 50]) < Bd * 3)[:, 0]
c = ax.pcolormesh(I[jj, :] * 180 / np.pi, D[jj, :] * 1e-3,
                  20 * np.log10(np.abs(G_ref[jj, :]) / np.max(np.abs(G_ref[jj, :]))), vmin=-50, vmax=20,
                  rasterized=True)
ax.set_ylabel('Doppler shift [kHz]')
ax.set_xlabel('incidence angle [deg]')
cbar = fig.colorbar(c, ax=ax, label='[dB]')
# Bd lines
ax.plot(np.array((incidence[0], incidence[-1])) * 180 / np.pi, np.ones(2) * Bd / 2e3, 'r')
ax.plot(np.array((incidence[0], incidence[-1])) * 180 / np.pi, -np.ones(2) * Bd / 2e3, 'r', label='Bd')
ax.legend()
fig.set_figwidth(fw)
fig.set_figheight(fh)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)
for t in (cbar.ax.get_yticklabels() + [cbar.ax.yaxis.label]):
    t.set_fontsize(fontsize)
plt.tight_layout()
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig("niceplotsforpaper/ref_on_doppler.svg", format="svg", dpi=600)
plt.show()

plt.show()
# %%

fig, ax = plt.subplots(1)
# slice the array to relevant bit
jj = np.argwhere(np.abs(D[:, 50]) < Bd / 1.8)[:, 0]
c = ax.pcolormesh(I[jj, :] * 180 / np.pi, D[jj, :], 20 * np.log10(np.abs(air[jj, :])),
                  rasterized=True)
fig.colorbar(c, ax=ax, label='[dB]')
ax.set_ylim(-Bd / 1.9, Bd / 1.9)
ax.set_ylabel('Doppler shift [Hz]')
ax.set_xlabel('incidence angle [deg]')
plt.show()

# %% cut
fig, ax = plt.subplots(1)
ax.plot(A[:, 50], np.abs(AIR[:, 50]) / np.max(AIR[:, 50]), label='distorted antenna')
ax.plot(A[:, 50], np.abs(AIRR[:, 50]), '--', label='reference')
ax.legend()
ax.set_xlabel('ground azimuth [m]')
ax.set_ylabel('normalized amplitude')
ax.set_xlim(-4, 4)
plt.show()

# %% cut but in dB


fig, ax = plt.subplots(1)
ax.plot(A[:, 50], 20 * np.log10(np.abs(AIR[:, 50]) / np.max(AIR[:, 50])), label='distorted antenna')
ax.plot(A[:, 50], 20 * np.log10(np.abs(AIRR[:, 50])), '--', label='reference')
ax.legend(loc='upper right', prop={"size": 8})
ax.set_xlabel('ground azimuth [m]')
ax.set_ylabel('normalized amplitude [dB]')
ax.set_xlim(-4, 4)
ax.set_ylim(-28, 0)

fig.set_figwidth(fw)
fig.set_figheight(fh)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)
plt.tight_layout()
plt.show()
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig("niceplotsforpaper/AIScut.svg", format="svg")
# %%
fig, ax = plt.subplots(1)
# slice the output for ease of plotting
ii = np.argwhere(np.abs(A[:, 50]) < 4)[:, 0]
c = ax.pcolormesh(I[ii, :] * 180 / np.pi, A[ii, :], 20 * np.log10(np.abs(AIR[ii, :]) / np.max(np.abs(AIR[ii, :]))),
                  rasterized=True)
cbar = fig.colorbar(c, ax=ax, label='[dB]')

ax.set_ylabel('ground azimuth [m]')
ax.set_xlabel('incidence angle [deg]')
fig.set_figwidth(fw)
fig.set_figheight(fh)
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(fontsize)
for t in (cbar.ax.get_yticklabels() + [cbar.ax.yaxis.label]):
    t.set_fontsize(fontsize)
plt.tight_layout()
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig("niceplotsforpaper/AIS.svg", format="svg", dpi=600)
plt.show()

# %%%
# last cell
print('prrrrrrr')
