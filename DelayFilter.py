import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class DelayFilter(object):
    def __init__(self, fs, delay, order=32, f_high=None):
        """
        fir delay filter
        Args:
            fs: sample frequency
            delay: time of delay to be implied, second
            order: order of filter
            f_high: #
        Returns:
             b, a: coefficients of delay filter
        """
        EPSILON = 1e-10
        t_unit = 1./fs
        if f_high is None:
            f_high = 0.9*fs/2.
        f_high_norm = f_high/fs

        T = 1./fs
        win_len = order*T  # Window duration (seconds)
        t = np.arange(-win_len/2, win_len/2+EPSILON, T)

        t_delayed = t - delay
        ir_filter = f_high_norm * (1 + np.cos(-2*np.pi*fs/order * t_delayed)) * np.sinc(2*f_high * t_delayed)
        self.b = ir_filter
        self.a = 1
        self.fs = fs
        self.delay = delay
        self.order = order

    def plot_spectrum(self):
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(self.b)
        spec = np.fft.fft(self.b)
        amp_spec, phase_spec = np.abs(spec), np.angle(spec)
        ax[1].plot(amp_spec)
        ax[2].plot(np.unwrap(phase_spec))

    def filter(self, x, is_padd=False):
        if is_padd:
            x_local = np.concatenate([x, np.zeros(self.order)])
        else:
            x_local = x
        y = scipy.signal.lfilter(self.b, self.a, x_local)
        return y


def test():
    import os
    plot_settings = {'linewidth': 2}

    delay_filter = DelayFilter(16000, 0.0005, 32)
    x = np.zeros(200)
    x[10] = 1
    y = delay_filter.filter(x)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, **plot_settings, label='origin')
    ax.plot(y, **plot_settings, label='delayed')
    plt.legend()
    os.makedirs('img', exist_ok=True)
    fig.savefig('img/delay_filter.png')
    plt.show()


if __name__ == '__main__':
    test()