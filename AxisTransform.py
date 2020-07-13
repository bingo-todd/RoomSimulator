import numpy as np


def cal_transform_matrix(angle_off):
    """
    Args:
         angle_off: azimuth shift in three plane [azi_off, ele_off, roll_off]
    """
    azi_c, ele_c, roll_c = np.cos(angle_off)
    azi_s, ele_s, roll_s = np.sin(angle_off)
    tm = np.array([
        [ele_c * azi_c, ele_c * azi_s, -ele_s],
        [roll_s * ele_s * azi_c - roll_c * azi_s,
         roll_s * ele_s * azi_s + roll_c * azi_c,
         roll_s * ele_c],
        [roll_c * ele_s * azi_c + roll_s * azi_s,
         roll_c * ele_s * azi_s - roll_s * azi_c,
         roll_c * ele_c]])
    return tm


