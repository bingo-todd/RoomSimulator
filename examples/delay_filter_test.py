import os
import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.append('../RoomSimulator')
from DelayFilter import DelayFilter


def test():
    plot_settings = {'linewidth': 2}

    delay_filter = DelayFilter(16000, 0.0001, 1024, f_high=0.9)
    x = np.zeros(1024)
    x[10] = 1
    y = delay_filter.filter(x)

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(8, 3))
    ax[0].plot(np.arange(1024), x, **plot_settings, label='origin')
    ax[0].plot(np.arange(1024), y, **plot_settings, label='delayed')
    ax[0].annotate('10', color='blue',
            xy=(10, x[10]), xycoords='data',
            xytext=(7, 0.9), 
            arrowprops=dict(facecolor='blue', arrowstyle='simple'),
            horizontalalignment='right', verticalalignment='top')
    ax[0].annotate('11.6', color='orange',
            xy=(11.6, (y[11]+y[12])/2.), xycoords='data',
            xytext=(14, 0.9), 
            arrowprops=dict(facecolor='orange', arrowstyle='simple'), 
            horizontalalignment='left', verticalalignment='top')
    ax[0].set_xlim((0, 50))
    ax[0].set_ylim((-0.25, 1.2))
    ax[0].set_xlabel('sample')
    ax[0].legend()
    ax[0].set_title('waveform')

    ax[1].plot(np.arange(1024)/1024, np.abs(np.fft.fft(x)), label='origin')
    ax[1].plot(np.arange(1024)/1024, np.abs(np.fft.fft(y)), label='delayed')
    ax[1].set_xlabel('normalized frequency') 
    ax[1].set_ylabel('dB') 
    ax[0].set_title('amplitude spectrum')

    fig.savefig('../images/delay_filter.png', dpi=100)


if __name__ == '__main__':
    test()
