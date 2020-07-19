import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_cube(cube_size, extra_point=None, extra_point_marker='+',
              orig_point=None, bright_color_all=None):
    if orig_point is None:
        orig_point = [0, 0, 0]
    point_all_unit_cube = np.asarray(
        [[0, 0, 0],
         [1, 0, 0],
         [1, 1, 0],
         [0, 1, 0],
         [0, 0, 1],
         [1, 0, 1],
         [1, 1, 1],
         [0, 1, 1]]
    )

    if bright_color_all is None:
        color_all = [(1, 0, 0, 0.1),  # blue
                     (1, 1, 0, 0.1),  # yellow
                     (0, 1, 1, 0.1),  # cyan
                     (1, 0, 1, 0.1),  # magenta
                     (0.5, 0.5, 0.5, 0.1),  # gray
                     (0, 0.5, 0, 0.1)]  # green
    else:
        cmap = matplotlib.cm.get_cmap('jet')
        color_all = []
        for i in range(6):
            color_all.append((*cmap(bright_color_all[i])[:3], 0.3))

    cube_size = np.asarray(cube_size)
    orig_point = np.asarray(orig_point)
    point_all = point_all_unit_cube * cube_size[np.newaxis, :]
    edge_all = [
        [point_all[0], point_all[1], point_all[2], point_all[3]],  # z=z_min
        [point_all[0], point_all[1], point_all[5], point_all[4]],  # y=x_min
        [point_all[1], point_all[2], point_all[6], point_all[5]],  # x=x_max
        [point_all[2], point_all[3], point_all[7], point_all[6]],  # y=y_max
        [point_all[3], point_all[0], point_all[4], point_all[7]],  # x=x_min
        [point_all[4], point_all[5], point_all[6], point_all[7]]   # z=z_max
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    face_all = Poly3DCollection(edge_all, linewidths=1, edgecolors='k')
    face_all.set_facecolor(color_all)
    ax.add_collection3d(face_all)

    # Plot the point_all themselves to force the scaling of the axes
    ax.scatter(point_all[:, 0], point_all[:, 1], point_all[:, 2], s=0)

    if extra_point is not None:
        dist_all = np.sqrt(np.sum(extra_point**2, axis=1))
        min_dist, max_dist = np.min(dist_all), np.max(dist_all)
        n_seg = 10
        dist_per_seg = (max_dist-min_dist)/n_seg
        for seg_i in range(n_seg):
            index = np.logical_and(dist_all>seg_i*dist_per_seg, dist_all<(seg_i+1)*dist_per_seg)
            ax.scatter(extra_point[index, 0], extra_point[index, 1], extra_point[index, 2],
                       marker=extra_point_marker, alpha=0.9-seg_i/n_seg*0.8)

    # ax.set_aspect('equal')
    axisEqual3D(ax)
    return fig, ax


if __name__ == '__main__':
    fig = plot_cube([4, 6, 8], bright_color_all=[0.1, 0.2, 0.5, 0.6, 0.1, 0.8])
    fig.savefig('img/plot_cube_test.png')
