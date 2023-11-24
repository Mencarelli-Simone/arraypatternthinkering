import mayavi.mlab as ml
import numpy as np
ml.figure(1)
ml.test_contour3d()
ml.figure(2)
ml.test_flow()
ml.figure(3)
ml.test_imshow()
ml.figure(4)
ml.test_mesh()
ml.figure(5)
ml.test_points3d()
ml.figure(6)
ml.test_quiver3d()
ml.figure(7)
ml.test_surf()
#%%
ml.show()
#%%
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010, Enthought
# License: BSD style

import numpy as np

# The number of points per line
N = 300

# The scalar parameter for each line
t = np.linspace(-2 * np.pi, 2 * np.pi, N)

from mayavi import mlab
mlab.figure(1, size=(400, 400), bgcolor=(0, 0, 0))
mlab.clf()

# We create a list of positions and connections, each describing a line.
# We will collapse them in one array before plotting.
x = list()
y = list()
z = list()
s = list()
connections = list()

# The index of the current point in the total amount of points
index = 0

# Create each line one after the other in a loop
for i in range(50):
    x.append(np.sin(t))
    y.append(np.cos((2 + .02 * i) * t))
    z.append(np.cos((3 + .02 * i) * t))
    s.append(t)
    # This is the tricky part: in a line, each point is connected
    # to the one following it. We have to express this with the indices
    # of the final set of points once all lines have been combined
    # together, this is why we need to keep track of the total number of
    # points already created (index)
    connections.append(np.vstack(
                       [np.arange(index,   index + N - 1.5),
                        np.arange(index + 1, index + N - .5)]
                            ).T)
    index += N

# Now collapse all positions, scalars and connections in big arrays
x = np.hstack(x)
y = np.hstack(y)
z = np.hstack(z)
s = np.hstack(s)
connections = np.vstack(connections)

# Create the points
src = mlab.pipeline.scalar_scatter(x, y, z, s)

# Connect them
src.mlab_source.dataset.lines = connections
src.update()

# The stripper filter cleans up connected lines
lines = mlab.pipeline.stripper(src)

# Finally, display the set of lines
mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)

# And choose a nice view
mlab.view(33.6, 106, 5.5, [0, 0, .05])
mlab.roll(125)
mlab.show()

#%%
import numpy as np
#set output to qt figures
#%matplotlib qt
a = np.random.random((4, 4))
from mayavi import mlab
mlab.surf(a)
mlab.show_pipeline()

#%%
mlab.show()

#%%
mlab.figure(1)
mlab.clf()
dx = 1
x = np.array([-dx / 2, dx / 2])
y = np.array([-dx / 2, dx / 2])
z = np.array([[0, 0], [0, 0]])


surf = mlab.pipeline.surface(src)

mlab.show()
#%%


import numpy as np
from mayavi.mlab import *
from numpy import pi, sin, cos, mgrid
def test_mesh():
    """A very pretty picture of spherical harmonics translated from
    the octaviz example."""
    pi = np.pi
    cos = np.cos
    sin = np.sin
    dphi, dtheta = pi / 250.0, pi / 250.0
    [phi, theta] = np.mgrid[0:pi + dphi * 1.5:dphi,
                            0:2 * pi + dtheta * 1.5:dtheta]
    m0 = 4
    m1 = 3
    m2 = 2
    m3 = 3
    m4 = 6
    m5 = 2
    m6 = 6
    m7 = 4
    r = sin(m0 * phi) ** m1 + cos(m2 * phi) ** m3 + \
        sin(m4 * theta) ** m5 + cos(m6 * theta) ** m7
    x = r * sin(phi) * cos(theta)
    y = r * cos(phi)
    z = r * sin(phi) * sin(theta)

    return mesh(x, y, z, colormap="bone")

a = test_mesh()
show_pipeline()