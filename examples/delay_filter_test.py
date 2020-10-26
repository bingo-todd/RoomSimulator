import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('../RoomSimulator')
from utils import delay_filter, delay_filter_fft  # noqa: E402


plot_settings = {'linewidth': 2}


def test(delay_filter_func, ax, n_sample):

    x = np.zeros(1024)
    x[10] = 1

    x_delayed = delay_filter_func(x, n_sample)

    ax[0].plot(np.arange(1024), x, **plot_settings, label='origin')
    ax[0].plot(np.arange(1024), x_delayed, **plot_settings, label='delayed')
    ax[0].set_title('waveform')
    ax[0].set_xlim((0, 50))
    ax[0].set_ylim((-0.25, 1.2))
    ax[0].set_xlabel('sample')
    ax[0].legend()
    # ax[1].set_ylabel('dB')

    ax[1].plot(np.arange(1024)/1024,
               np.abs(np.fft.fft(x)),
               label='origin')
    ax[1].plot(np.arange(1024)/1024,
               np.abs(np.fft.fft(x_delayed)),
               label='delayed')
    ax[1].set_ylim([-2, 2])
    ax[1].set_title('amplitude spectrum')
    ax[1].set_xlabel('normalized frequency')


if __name__ == '__main__':
    fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(8, 6))
    test(delay_filter, ax[0], 0.6)
    ax[0, 0].set_ylabel('v1')
    test(delay_filter_fft, ax[1], 0.6)
    ax[1, 0].set_ylabel('v2')
    fig.savefig('../images/delay_filter.png', dpi=100)
