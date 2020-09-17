import numpy as np


def rotate2tm(rotate):
    """
    Args:
     rotate:[pitch, yaw, roll] counterclockwise rotation angles
           pitch: rotation angle about x-axis
           roll: rotation angle of about y-axis
           yaw: rotation angle of about z-axis
    """
    rotate_rad = np.asarray(rotate)/180*np.pi
    pitch_c, yaw_c, roll_c = np.cos(rotate_rad)
    pitch_s, yaw_s, roll_s = np.sin(rotate_rad)
    tm = np.array(
         [[roll_c*yaw_c,     roll_c*yaw_s* pitch_s-roll_s*pitch_c, roll_c*yaw_s*pitch_c+roll_s*pitch_s],
          [roll_s*yaw_c,     roll_s*yaw_s*pitch_s+roll_c*pitch_c,  roll_s*yaw_s*pitch_c-roll_c*pitch_s],   
          [-yaw_s,           yaw_c*pitch_s,                        yaw_c*pitch_c]])
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
     plot_ax(ax, tm)
     ax.scatter(*pos_orig)

     # rotate poin according to tm
     fig2 = plt.figure()
     tm = rotate2tm([0, 0, 90])
     ax2 = fig2.add_subplot(111, projection='3d')
     plot_ax(ax2, tm)
     pos_rotate_point = np.matmul(tm, pos_orig)
     ax2.scatter(*pos_rotate_point)

     plt.show()

