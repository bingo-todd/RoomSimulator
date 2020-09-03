import numpy as np
from scipy import io as sio
import matplotlib.pyplot as plt


def main():
    rir_matlab = sio.loadmat('H_ROOM_MIT_S1.mat', squeeze_me=True)['data']

    rir_python = np.load('rir.npy')

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].plot(rir_matlab[:, 0])
    ax[0].plot(rir_python[:, 0])
    ax[1].plot(rir_matlab[:, 1])
    ax[1].plot(rir_python[:, 1])
    plt.show()



if __name__ == '__main__':
    main()