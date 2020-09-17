import numpy as np


def rotate2tm(rotate):
    """
    Args:
     rotate:[roll, pitch, yaw] counterclockwise rotation angles
           roll: rotation angle about x-axis
           pitch: rotation angle of about y-axis
           yaw: rotation angle of about z-axis
    """
    rotate_rad = np.asarray(rotate)/180*np.pi
    roll_c, pitch_c, yaw_c = np.cos(rotate_rad)
    roll_s, pitch_s, yaw_s = np.sin(rotate_rad)
    tm = np.array(
         [[pitch_c*yaw_c,                       -pitch_c*yaw_s,                     pitch_s],
          [roll_c*yaw_s+yaw_c*roll_s*pitch_s,   roll_c*yaw_c-roll_s*pitch_s*yaw_s,  -pitch_c*roll_s],   
          [roll_s*yaw_s,                        yaw_c*roll_s+roll_c*pitch_s*yaw_s,  roll_c*pitch_c]])
    return tm


def plot_ax(ax, tm):
     x_arrow_plot_settings = {
          'arrow_length_ratio': 0.4,
          'pivot': 'tail',
          'color': [1, 20./255, 147./255],
          'linewidth': 2}
     y_arrow_plot_settings = {
          'arrow_length_ratio': 0.4,
          'pivot': 'tail',
          'color': [155./255, 48./255, 1],
          'linewidth': 2}
     z_arrow_plot_settings = {
          'arrow_length_ratio': 0.4,
          'pivot': 'tail',
          'color': [1, 165./255, 0],
          'linewidth': 2}

     ax.quiver(*[0, 0, 0], *tm[:, 0], **x_arrow_plot_settings, label='x')
     ax.quiver(*[0, 0, 0], *tm[:, 1], **y_arrow_plot_settings, label='y')
     ax.quiver(*[0, 0, 0], *tm[:, 2], **z_arrow_plot_settings, label='z')
     ax.legend()

     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.set_zlabel('z')


if __name__ == "__main__":
     import matplotlib.pyplot as plt 

     pos_orig = np.asarray([4, 4, 4])

     fig = plt.figure()
     tm = rotate2tm([0, 0, 0])
     ax = fig.add_subplot(111, projection='3d')
     ax.scatter(0, 0, 0, label='o')
     # plot_ax(ax, tm)
     ax.scatter(*pos_orig, label='origin')

     tm_z = rotate2tm([0, 0, 30])
     pos_rotate_point = np.matmul(tm_z, pos_orig)
     ax.scatter(*pos_rotate_point, label='z')

     tm_y = rotate2tm([0, 30, 0])
     pos_rotate_point = np.matmul(tm_y, pos_orig)
     ax.scatter(*pos_rotate_point, label='y')

     tm_x = rotate2tm([30, 0, 0])
     pos_rotate_point = np.matmul(tm_x, pos_orig)
     ax.scatter(*pos_rotate_point, label='x')

     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.legend()

     plt.show()

