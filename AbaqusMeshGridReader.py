# functions to import the mechanical meshgrid from abaqus txt files

# %% includes
import numpy as np
from scipy.interpolate import griddata
import mayavi.mlab as ml
import copy


# %% Functions
def readmesh(filename: str, rows, columns, xpoints, ypoints):
    """

    :param filename:
    :param rows: rows in the table
    :param columns: columns in the table
    :param xpoints: x size of meshgrid (width of the shell)
    :param ypoints: y size of meshgrid (length of the shell)
    :return: Xundef, Yundef, Zundef, Xdef, Ydef, Zdef meshgrids
    """
    # %% reader
    with open(filename, 'r') as file:
        content = file.read()
    content = content.split('\n')

    dataMatrix = np.zeros((rows, columns))

    for i in range(len(content)):
        if "Freq" in content[i]:
            for j in range(rows):
                dataline = content[i + j + 1]
                dataline = dataline.split(" ")
                if len(dataline) < columns:  # eof
                    print('eof')
                    break
                else:
                    while "" in dataline:
                        dataline.remove("")
                    dataline = np.array(dataline[0:columns], dtype=float).reshape((1, columns))
                    dataMatrix[j, :] = dataline.astype('float')
            print('eof')
    # %%
    # raw line vectors
    Xundef = dataMatrix[:, 2]
    Yundef = dataMatrix[:, 3]
    Zundef = dataMatrix[:, 4]
    Xdef = dataMatrix[:, 5]
    Ydef = dataMatrix[:, 6]
    Zdef = dataMatrix[:, 7]
    # %% reconstructing the meshgrid

    Xundef = Xundef.reshape((ypoints, xpoints))  # actually y
    Yundef = Yundef.reshape((ypoints, xpoints))  # actually z
    Zundef = Zundef.reshape((ypoints, xpoints))  # actually x
    Xdef = Xdef.reshape((ypoints, xpoints))
    Ydef = Ydef.reshape((ypoints, xpoints))
    Zdef = Zdef.reshape((ypoints, xpoints))
    return Xundef, Yundef, Zundef, Xdef, Ydef, Zdef


def orthogonalizer_same_median(hbx, hby, hbz, vbx, vby, vbz):
    """
    orthogonalize two vectors to the closest orthogonal basis (same plane) sharing the same median vector
    :param hbx: horizontal vector x component
    :param hby: horizontal vector y component
    :param hbz: horizontal vector z component
    :param vbx: vertical vector x component
    :param vby: vertical vector y component
    :param vbz: vertical vector z component
    :return: x1, x2, x3, y1, y2, y3, z1, z2, z3
    """

    # 4 find the median vector between the two basis vectors
    mx = (hbx + vbx) / 2
    my = (hby + vby) / 2
    mz = (hbz + vbz) / 2
    # 5. define two independent unitary vectors from v and m, these will be the starting point
    # basis for a Gram-Dhmidt orthonormalization procedure, grom now on the xyz components of
    # each vector are denoted numerically 123, and the vectors are normalized to always be unitary
    # 5.1. median versor
    m1 = mx / np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    m2 = my / np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    m3 = mz / np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    # 5.2. horizontal versor
    h1 = hbx / np.sqrt(hbx ** 2 + hby ** 2 + hbz ** 2)
    h2 = hby / np.sqrt(hbx ** 2 + hby ** 2 + hbz ** 2)
    h3 = hbz / np.sqrt(hbx ** 2 + hby ** 2 + hbz ** 2)
    # 6. Gram-Schmidt orthonormalization to find n, i.e., the vector orthogonal to m and
    # closest to v, belonging to the same span<m,v>
    # 6.1 scalar projection of v on m
    v1m = h1 * m1 + h2 * m2 + h3 * m3
    # 6.2 orthogonalization
    n1 = h1 - v1m * m1
    n2 = h2 - v1m * m2
    n3 = h3 - v1m * m3
    # 6.3 normalization
    # modulus of n
    nabs = np.sqrt(n1 ** 2 + n2 ** 2 + n3 ** 2)
    n1 = n1 / nabs
    n2 = n2 / nabs
    n3 = n3 / nabs
    # 7. Find the z versor (normal to the rotation plane) as the cross product of m and n
    # to form a right-handed orthonormal basis
    z1 = m2 * n3 - m3 * n2
    z2 = m3 * n1 - m1 * n3
    z3 = m1 * n2 - m2 * n1
    # 8. rotate m and n of 45 degrees to find the x and y versors
    # 8.1 x versor
    x1 = m1 / np.sqrt(2) - n1 / np.sqrt(2)
    x2 = m2 / np.sqrt(2) - n2 / np.sqrt(2)
    x3 = m3 / np.sqrt(2) - n3 / np.sqrt(2)
    # 8.2 y versor
    y1 = m1 / np.sqrt(2) + n1 / np.sqrt(2)
    y2 = m2 / np.sqrt(2) + n2 / np.sqrt(2)
    y3 = m3 / np.sqrt(2) + n3 / np.sqrt(2)
    # 9. return everything
    return x1, x2, x3, y1, y2, y3, z1, z2, z3


# %% Classes
class AbaqusMesh:
    def __init__(self, filename: str, rows, columns, xpoints, ypoints):
        """

        :param filename:
        :param rows: rows in the table
        :param columns: columns in the table
        :param xpoints: x size of meshgrid
        :param ypoints: y size of meshgrid
        """

        self.filename = filename
        self.rows = rows
        self.columns = columns
        self.xpoints = xpoints
        self.ypoints = ypoints
        self.Xundef, self.Yundef, self.Zundef, self.Xdef, self.Ydef, self.Zdef = readmesh(filename, rows, columns,
                                                                                          xpoints, ypoints)
        self.defZ = self.Zdef - self.Zundef
        self.defY = self.Ydef - self.Yundef
        self.defX = self.Xdef - self.Xundef
        # lcs of the cells undeformed
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.z1 = None
        self.z2 = None
        self.z3 = None
        # lcs of the cells deformed
        self.x1_def = None
        self.x2_def = None
        self.x3_def = None
        self.y1_def = None
        self.y2_def = None
        self.y3_def = None
        self.z1_def = None
        self.z2_def = None
        self.z3_def = None
        # cell centers
        self.Xcell = None
        self.Ycell = None
        self.Zcell = None
        self.Xcell_def = None
        self.Ycell_def = None
        self.Zcell_def = None

    def scale(self, xscale=1, yscale=1, zscale=1):
        """
        scale the deformation in one or multiple axis
        :param xscale:
        :param yscale:
        :param zscale:
        :return:
        """
        self.Ydef = self.Yundef + self.defY * yscale
        self.Xdef = self.Xundef + self.defX * xscale
        self.Zdef = self.Zundef + self.defZ * zscale

    # symmetries
    def symmetric_def(self, axis='x'):
        """
        deform the mesh symmetrically on one axis
        resets the scaling factor
        :param axis:
        :return:
        """
        if axis == 'x':
            self.defX = -self.defX
            self.Xdef = self.Xundef + self.defX
        elif axis == 'y':
            self.defY = -self.defY
            self.Ydef = self.Yundef + self.defY
        elif axis == 'z':
            self.defZ = -self.defZ
            self.Zdef = self.Zundef + self.defZ

    def get_cell_centers(self):
        """
        get the cell centers of the meshgrid in the undeformed configuration and the deformed one
        :return:
        """
        # average of 4 corner points
        self.Xcell = (self.Xundef[:-1, :-1] + self.Xundef[1:, :-1] + self.Xundef[:-1, 1:] + self.Xundef[1:, 1:]) / 4
        self.Ycell = (self.Yundef[:-1, :-1] + self.Yundef[1:, :-1] + self.Yundef[:-1, 1:] + self.Yundef[1:, 1:]) / 4
        self.Zcell = (self.Zundef[:-1, :-1] + self.Zundef[1:, :-1] + self.Zundef[:-1, 1:] + self.Zundef[1:, 1:]) / 4
        self.Xcell_def = (self.Xdef[:-1, :-1] + self.Xdef[1:, :-1] + self.Xdef[:-1, 1:] + self.Xdef[1:, 1:]) / 4
        self.Ycell_def = (self.Ydef[:-1, :-1] + self.Ydef[1:, :-1] + self.Ydef[:-1, 1:] + self.Ydef[1:, 1:]) / 4
        self.Zcell_def = (self.Zdef[:-1, :-1] + self.Zdef[1:, :-1] + self.Zdef[:-1, 1:] + self.Zdef[1:, 1:]) / 4
        return self.Xcell, self.Ycell, self.Zcell, self.Xcell_def, self.Ycell_def, self.Zcell_def

    def get_cell_lcs(self):
        """
        finds an orthonormal basis to match the rotation of the element cell (without deformation)
        a trapezoid will be approximated to a rotated element square cell
        :return:
        """
        # I undeformed mesh
        # 1. find the horizontal versors
        hx = self.Xundef[:, 1:] - self.Xundef[:, :-1]
        hy = self.Yundef[:, 1:] - self.Yundef[:, :-1]
        hz = self.Zundef[:, 1:] - self.Zundef[:, :-1]
        # 2. find the vertical versors
        vx = self.Xundef[1:, :] - self.Xundef[:-1, :]
        vy = self.Yundef[1:, :] - self.Yundef[:-1, :]
        vz = self.Zundef[1:, :] - self.Zundef[:-1, :]
        # 3. average two by two to find the cell basis vectors (rotation plane)
        hbx = (hx[:-1, :] + hx[1:, :]) / 2
        hby = (hy[:-1, :] + hy[1:, :]) / 2
        hbz = (hz[:-1, :] + hz[1:, :]) / 2
        vbx = (vx[:, :-1] + vx[:, 1:]) / 2
        vby = (vy[:, :-1] + vy[:, 1:]) / 2
        vbz = (vz[:, :-1] + vz[:, 1:]) / 2
        x1, x2, x3, y1, y2, y3, z1, z2, z3 = orthogonalizer_same_median(hbx, hby, hbz, vbx, vby, vbz)
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        # II deformed mesh
        # 1. find the horizontal versors
        hx = self.Xdef[:, 1:] - self.Xdef[:, :-1]
        hy = self.Ydef[:, 1:] - self.Ydef[:, :-1]
        hz = self.Zdef[:, 1:] - self.Zdef[:, :-1]
        # 2. find the vertical versors
        vx = self.Xdef[1:, :] - self.Xdef[:-1, :]
        vy = self.Ydef[1:, :] - self.Ydef[:-1, :]
        vz = self.Zdef[1:, :] - self.Zdef[:-1, :]
        # 3. average two by two to find the cell basis vectors (rotation plane)
        hbx = (hx[:-1, :] + hx[1:, :]) / 2
        hby = (hy[:-1, :] + hy[1:, :]) / 2
        hbz = (hz[:-1, :] + hz[1:, :]) / 2
        vbx = (vx[:, :-1] + vx[:, 1:]) / 2
        vby = (vy[:, :-1] + vy[:, 1:]) / 2
        vbz = (vz[:, :-1] + vz[:, 1:]) / 2
        x1, x2, x3, y1, y2, y3, z1, z2, z3 = orthogonalizer_same_median(hbx, hby, hbz, vbx, vby, vbz)
        self.x1_def = x1
        self.x2_def = x2
        self.x3_def = x3
        self.y1_def = y1
        self.y2_def = y2
        self.y3_def = y3
        self.z1_def = z1
        self.z2_def = z2
        self.z3_def = z3
        return self.x1, self.x2, self.x3, self.y1, self.y2, self.y3, self.z1, self.z2, self.z3, \
            self.x1_def, self.x2_def, self.x3_def, self.y1_def, self.y2_def, self.y3_def, self.z1_def, \
            self.z2_def, self.z3_def

    ## graphics
    def plot_nominal(self, **kwargs):
        ml.mesh(self.Xundef, self.Yundef, self.Zundef, **kwargs)

    def plot_deformed(self, **kwargs):
        absdef = np.sqrt(self.defX ** 2 + self.defY ** 2 + self.defZ ** 2)
        ml.mesh(self.Xdef, self.Ydef, self.Zdef, scalars=absdef, **kwargs)

    def plot_cell_centers(self, **kwargs):
        self.get_cell_centers()
        ml.points3d(self.Xcell, self.Ycell, self.Zcell, **kwargs)
        ml.points3d(self.Xcell_def, self.Ycell_def, self.Zcell_def, **kwargs)

    def plot_lcs(self, **kwargs):
        self.get_cell_lcs()
        ml.quiver3d(self.Xcell, self.Ycell, self.Zcell, self.x1, self.x2, self.x3, color=(1, 0, 0), **kwargs)
        ml.quiver3d(self.Xcell, self.Ycell, self.Zcell, self.y1, self.y2, self.y3, color=(0, 1, 0), **kwargs)
        ml.quiver3d(self.Xcell, self.Ycell, self.Zcell, self.z1, self.z2, self.z3, color=(0, 0, 1), **kwargs)
        ml.quiver3d(self.Xcell_def, self.Ycell_def, self.Zcell_def, self.x1_def, self.x2_def, self.x3_def,
                    color=(1, 0, 0), **kwargs)
        ml.quiver3d(self.Xcell_def, self.Ycell_def, self.Zcell_def, self.y1_def, self.y2_def, self.y3_def,
                    color=(0, 1, 0), **kwargs)
        ml.quiver3d(self.Xcell_def, self.Ycell_def, self.Zcell_def, self.z1_def, self.z2_def, self.z3_def,
                    color=(0, 0, 1), **kwargs)


# %% main
if __name__ == "__main__":
    filename = 'random_analysis_results/res_2mode.txt'
    rows = 3312
    columns = 11
    xpoints = 23
    ypoints = 144
    defomesh = AbaqusMesh(filename, rows, columns, xpoints, ypoints)
    defomesh.scale(yscale=90)
    # %% plotter
    ml.figure(bgcolor=(0, 0, 0))
    defomesh.plot_nominal(opacity=1, color=(.8, .8, .8))
    defomesh.plot_deformed(colormap='jet', opacity=0.99)
    defomesh.plot_deformed(color=(0, 0, 1), representation='wireframe', opacity=0.99)
    defomesh1 = copy.deepcopy(defomesh)

    # %% symmetries
    defomesh1.symmetric_def(axis='y')
    defomesh1.scale(yscale=90)
    defomesh1.plot_deformed(colormap='jet', opacity=0.99)
    defomesh1.plot_deformed(color=(1, 0, 0), representation='wireframe', opacity=0.99)
    defomesh1.plot_cell_centers(color=(1, 1, 1), scale_factor=0.005)

    # lcs
    defomesh1.plot_lcs(scale_factor=0.01, mode='arrow')
    ml.show()
    x1, x2, x3, y1, y2, y3, z1, z2, z3, x1_def, x2_def, x3_def, y1_def, y2_def, y3_def, z1_def, z2_def, z3_def = defomesh1.get_cell_lcs()
    # test check orthogonality of the basis deformed
    cp1 = x1_def * y1_def + x2_def * y2_def + x3_def * y3_def
    cp2 = x1_def * z1_def + x2_def * z2_def + x3_def * z3_def
    cp3 = y1_def * z1_def + y2_def * z2_def + y3_def * z3_def
    print(cp1, cp2, cp3)
    print(np.max(cp1))
    print(np.max(cp2))
    print(np.max(cp3))
    print(np.min(cp1))
    print(np.min(cp2))
    print(np.min(cp3))
