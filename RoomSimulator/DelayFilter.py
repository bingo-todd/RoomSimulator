import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def delay_filter(x, fs, delay, order=128, is_padd=False, padd_len=None,
                 f_high=None, id_padd=False):
        """
        fir delay filter
        Args:
            fs: sample frequency
            delay: time of delay to be implied, second
            order: order of filter
            f_high: normalized frequency
        """
        if f_high is None:
            f_high = 0.9

        t_delayed = np.arange(order) - delay*fs
        ir = (((f_high*order*np.sinc(f_high*t_delayed)
                * np.cos(np.pi/order*t_delayed))
               + np.cos(f_high*np.pi*t_delayed))
              * f_high*order/(order*order)/f_high)

        if is_padd:
            if padd_len is None:
                padd_len = order
            x = np.concatenate([x, np.zeros(padd_len)])
        x_delayed = scipy.signal.lfilter(ir, 1, x)
        return x_delayed


def delay_filter_fft(x, n_sample):
    """
    delay filter by modifying the phase spectrum
    """
    x_len = x.shape[0]
    x_spec = np.fft.fft(x)
    norm_freqs = np.arange(x_len)/x_len
    phase_shift = np.exp(1j*2*np.pi*norm_freqs*n_sample)
    x_spec_delayed = x_spec*phase_shift
    x_delayed = np.real(np.fft.ifft(x_spec_delayed))
    return x_delayed
