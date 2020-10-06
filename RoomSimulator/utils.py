import time
import os
import numpy as np
import scipy.signal
from BasicTools.get_file_path import get_file_path


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
    pos = np.asarray(
        [dist_xy*np.cos(angle_rad[0]),
         dist_xy*np.sin(angle_rad[0]),
         dist*np.sin(angle_rad[1])])
    return pos


def cartesian2pole(pos):
    dist = cal_dist(pos)
    dist_xy = np.sqrt(np.sum(pos[0:2]**2))
    # -180 ~ 180, right is positive, left is negative
    azi = -np.sign(pos[1]) * np.arccos(pos[0]/dist_xy)/rad_per_degree
    ele = np.arctan(pos[2]/dist_xy)/rad_per_degree
    return [azi, ele, dist]


def nonedelay_filter(b, a, x):
    if len(b.shape) == 0:
        b = b.reshape(1)
    return scipy.signal.filtfilt(b, a, x)


def norm_filter(b, a, x):
    if len(b.shape) == 0:
        b = b.reshape(1)
    return scipy.signal.lfilter(b, a, x)


class Logger(object):
    def __init__(self, log_path):
        self.logger = open(log_path, 'w')

    def info(self, log_str):
        self.logger.write(f'{time.ctime()} {log_str} \n')

    def warning(self, log_str):
        self.logger.write(f'{time.ctime()} {log_str} \n')

    def close(self):
        self.logger.close()


def delay_filter(x, n_sample, order=128, is_padd=False, padd_len=None,
                 f_high=None, id_padd=False):
    """
    fir delay filter
    """
    if f_high is None:
        f_high = 0.98

    n_sample_int = np.int(np.round(n_sample))
    if n_sample_int < 0:
        x = np.pad(x[n_sample_int:], [0, n_sample_int])
    elif n_sample_int > 0:
        x = np.pad(x[:-n_sample_int], [n_sample_int, 0])
    n_sample = n_sample - n_sample_int

    pid = os.getpid()
    ir_dir = f'dump/delay_filter/{pid}'
    if np.abs(n_sample) > 1e-10:
        ir_path = f'{ir_dir}/{n_sample:.5f}.npy'
        if not os.path.exists(ir_path):
            ir_path_all = get_file_path('dump/delay_filter',
                                        suffix='.npy',
                                        is_absolute=True)
            if len(ir_path_all) > 0:
                os.system(f"cp {' '.join(ir_path_all)} {ir_dir}")
        os.makedirs(f'dump/delay_filter_{pid}', exist_ok=True)
        if os.path.exists(ir_path):
            ir = np.load(ir_path, allow_pickle=True)
        else:
            sample_index_all = np.arange(order) - n_sample
            ir = (((f_high*order*np.sinc(f_high*sample_index_all)
                    * np.cos(np.pi/order*sample_index_all))
                   + np.cos(f_high*np.pi*sample_index_all))
                  * f_high*order/(order*order)/f_high)
            np.save(ir_path, ir)

        if is_padd:
            if padd_len is None:
                padd_len = order
            x = np.pad(x, [0, padd_len])
        x = scipy.signal.lfilter(ir, 1, x)
    return x


def delay_filter_fft(x, n_sample):
    """
    delay filter by modifying the phase spectrum
    """
    n_sample_int = np.int(np.round(n_sample))
    if n_sample_int < 0:
        x = np.pad(x[n_sample_int:], [0, n_sample_int])
    elif n_sample_int > 0:
        x = np.pad(x[:-n_sample_int], [n_sample_int, 0])
    n_sample = n_sample - n_sample_int

    if np.abs(n_sample) > 1e-10:
        x_len = x.shape[0]
        x_spec = np.fft.fft(x)
        norm_freqs = np.arange(x_len)/x_len
        phase_shift = np.exp(-1j*2*np.pi*norm_freqs*n_sample)
        x_spec_delayed = x_spec*phase_shift
        x = np.real(np.fft.ifft(x_spec_delayed))
    return x
