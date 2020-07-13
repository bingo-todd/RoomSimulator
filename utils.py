import numpy as np
import scipy.signal


def cal_dist(x1, x2=None):
    if x2 is None:
        x2 = np.zeros(3)
    return np.sqrt(np.sum((x1 - x2) ** 2))


def pole2cartesian(angle, dist=1):
    """
    convert
    """
    dist_xy = dist*np.cos(angle[2])
    pos = np.asarray([dist_xy*np.cos(angle[0]), dist_xy*np.sin(angle[1]), dist*np.sin(angle[2])])
    return pos


def cartesian2pole(pos):
    dist = cal_dist(pos)
    dist_xy = np.sqrt(np.sum(pos[0:2]**2))
    azi = np.arccos(pos[0]/dist_xy)
    ele = np.arcsin(pos[1]/dist_xy)
    return [azi, ele, dist]


def filter(b, a, x):
    if len(b.shape) == 0:
        b = b.reshape(1)
    return scipy.signal.lfilter(b, a, x)