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
        # 5.2. vertical versor
        v1 = vbx / np.sqrt(vbx ** 2 + vby ** 2 + vbz ** 2)
        v2 = vby / np.sqrt(vbx ** 2 + vby ** 2 + vbz ** 2)
        v3 = vbz / np.sqrt(vbx ** 2 + vby ** 2 + vbz ** 2)
        # 6. Gram-Schmidt orthonormalization to find n, i.e., the vector orthogonal to m and
        # closest to v, belonging to the same span<m,v>
        # 6.1 scalar projection of v on m
        v1m = v1 * m1 + v2 * m2 + v3 * m3
        # 6.2 orthogonalization
        n1 = v1 - v1m * m1
        n2 = v2 - v1m * m2
        n3 = v3 - v1m * m3
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
        # 9. reshape the vectors to match teh format required by the array class

        pass


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


# %% main
if __name__ == "__main__":
    filename = 'random_analysis_results/res_2mode.txt'
    rows = 3312
    columns = 11
    xpoints = 23
    ypoints = 144
    defomesh = AbaqusMesh(filename, rows, columns, xpoints, ypoints)
    defomesh.scale(yscale=20)
    # %% plotter
    ml.figure(bgcolor=(0, 0, 0))
    defomesh.plot_nominal(opacity=1, color=(.8, .8, .8))
    defomesh.plot_deformed(colormap='jet', opacity=0.99)
    defomesh.plot_deformed(color=(0, 0, 1), representation='wireframe', opacity=0.99)
    defomesh1 = copy.deepcopy(defomesh)

    # %% symmetries
    defomesh1.symmetric_def(axis='y')
    defomesh1.scale(yscale=20)
    defomesh1.plot_deformed(colormap='jet', opacity=0.99)
    defomesh1.plot_deformed(color=(1, 0, 0), representation='wireframe', opacity=0.99)
    defomesh1.plot_cell_centers(color=(1, 1, 1), scale_factor=0.005)
    ml.show()
