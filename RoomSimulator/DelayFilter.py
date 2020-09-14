import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class DelayFilter(object):
    def __init__(self, fs, delay, order=128, padd_len=None, f_high=None):
        """
        fir delay filter
        Args:
            fs: sample frequency
            delay: time of delay to be implied, second
            order: order of filter
            f_high: normalized frequency 
        Returns:
             b, a: coefficients of delay filter
        """
        if f_high is None:
            f_high = 0.9

        t_delayed = np.arange(order) - delay*fs
        ir = (f_high*order*np.sinc(f_high*t_delayed)*np.cos(np.pi/order*t_delayed) + np.cos(f_high*np.pi*t_delayed))*f_high*order/(order*order)/f_high

        self.b = ir
        self.a = 1
        self.fs = fs
        self.delay = delay
        self.order = order

        if padd_len is None:
            padd_len = order
        self.padd_len = padd_len

    def plot_spectrum(self):
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(self.b)
        spec = np.fft.fft(self.b)
        amp_spec, phase_spec = np.abs(spec), np.angle(spec)
        ax[1].plot(amp_spec)
        ax[2].plot(np.unwrap(phase_spec))
        return fig, ax

    def filter(self, x, is_padd=False):
        if is_padd:
            x_local = np.concatenate([x, np.zeros(self.padd_len)])
        else:
            x_local = x
        y = scipy.signal.lfilter(self.b, self.a, x_local)
        return y


