import time
import numpy as np
import scipy.signal
import logging
import os


rad_per_degree = np.pi/180


def cal_dist(x1, x2=None):
    if x2 is None:
        x2 = np.zeros(3)
    return np.sqrt(np.sum((x1 - x2) ** 2))


def pole2cartesian(angle_degree, dist=1):
    """
    convert
    """
    dist_xy = dist*np.cos(angle_degree[1])
    angle_rad = np.asarray(angle_degree)*rad_per_degree
    pos = np.asarray([dist_xy*np.cos(angle_rad[0]), dist_xy*np.sin(angle_rad[0]), dist*np.sin(angle_rad[1])])
    return pos


def cartesian2pole(pos):
    dist = cal_dist(pos)
    dist_xy = np.sqrt(np.sum(pos[0:2]**2))
    azi = -np.sign(pos[1]) * np.arccos(pos[0]/dist_xy)/rad_per_degree  # -180 ~ 180, right is positive, left is negative
    ele = np.arctan(pos[2]/dist_xy)/rad_per_degree  
    return [azi, ele, dist]


def filter(b, a, x):
    if len(b.shape) == 0:
        b = b.reshape(1)
    return scipy.signal.lfilter(b, a, x)


class My_Logger(object):
    def __init__(self, log_path):
        self.logger = open(log_path, 'w')

    def info(self, log_str):
        self.logger.write(f'{time.ctime()} {log_str} \n')
    
    def warning(self, log_str):
        self.logger.write(f'{time.ctime()} {log_str} \n')
    
    def close(self):
        self.logger.close()