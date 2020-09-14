import numpy as np
from scipy import io as sio
import matplotlib.pyplot as plt


def main():
    rir_matlab = sio.loadmat('H_ROOM_MIT_S1.mat', squeeze_me=True)['data']

    rir_python = np.load('rir.npy')

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, 
            tight_layout=True, figsize=(8, 3))
    ax[0].plot(rir_matlab[:, 0], label='roomsim')
    ax[0].plot(rir_python[:, 0], label='roomsimulator')
    ax[0].set_title('left ear')
    ax[0].legend()
    ax[1].plot(rir_matlab[:, 1])
    ax[1].plot(rir_python[:, 1])
    ax[1].set_title('right ear')
    fig.savefig('../images/validation.png', dpi=100)



if __name__ == '__main__':
    main()
