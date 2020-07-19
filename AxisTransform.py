import numpy as np


def view2tm(view):
    """
    [psi, phi, theta]
    +ve psi_rad is the yaw angle in the xy plane clockwise from the positive x axis (slew left).
    +ve theta_rad is the pitch angle from the xy plane (nose up).1	
    +ve phi_rad is roll clockwise about the +ve x axis (right wing dips).
    """
    view_rad = np.asarray(view)/180*np.pi
    psi_c, phi_c, theta_c = np.cos(view_rad)
    psi_s, phi_s, theta_s = np.sin(view_rad)
    tm = np.array([
        [theta_c * psi_c, theta_c * psi_s, -theta_s],
        [-phi_s * theta_s * psi_c - phi_c * psi_s,
         -phi_s * theta_s * psi_s + phi_c * psi_c,
         -phi_s * theta_c],
        [phi_c * theta_s * psi_c - phi_s * psi_s,
                phi_c * theta_s * psi_s + phi_s * psi_c,
         phi_c * theta_c]])
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

     ax.quiver(*[0, 0, 0], *tm[:, 0], **x_arrow_plot_settings)
     ax.quiver(*[0, 0, 0], *tm[:, 1], **y_arrow_plot_settings)
     ax.quiver(*[0, 0, 0], *tm[:, 2], **z_arrow_plot_settings)

     ax.set_xlabel('x')
     ax.set_ylabel('y')
     ax.set_zlabel('z')


if __name__ == "__main__":
     import matplotlib.pyplot as plt 

     pos_orig = np.asarray([4, 4, 4])

     fig = plt.figure()
     tm = view2tm([0, 0, 0])
     ax = fig.add_subplot(111, projection='3d')
     plot_ax(ax, tm)
     ax.scatter(*pos_orig)

     # rotate poin according to tm
     fig2 = plt.figure()
     tm = view2tm([90, 0, 0])
     ax2 = fig2.add_subplot(111, projection='3d')
     plot_ax(ax2, tm)
     pos_rotate_point = np.matmul(tm, pos_orig)
     ax2.scatter(*pos_rotate_point)


     # calculate new position of the origin point in the rotated axis
     fig3 = plt.figure()
     tm = view2tm([90, 0, 0])
     ax3 = fig3.add_subplot(111, projection='3d')
     plot_ax(ax3, tm)
     pos_rotate_axis = np.matmul(tm.T, pos_orig)
     ax3.scatter(*pos_rotate_axis)

     plt.show()

