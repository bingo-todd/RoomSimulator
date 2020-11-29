import numpy as np
from scipy import io as sio
import matplotlib.pyplot as plt
import configparser

from RoomSimulator import RoomSimulator


if __name__ == '__main__':

    room_config = configparser.ConfigParser()
    room_config['Room'] = {
        'size': '4, 4, 4',
        'RT60': ', '.join([f'{item}' for item in np.ones(6) * 0.2]),
        'A': '',
        'Fs': 44100,
        'reflect_order': -1,
        'HP_cutoff': 100}

    receiver_config = configparser.ConfigParser()
    receiver_config['Receiver'] = {
        'pos': '2, 2, 2',
        'rotate': '0, 0, 0',
        'n_mic': '2'}
    head_r = 0.145/2
    receiver_config['Mic_0'] = {
        'pos': f'0, {head_r}, 0',
        'rotate': '0, 0, 90',
        'direct_type': 'binaural_L'}
    receiver_config['Mic_1'] = {
        'pos': f'0, {-head_r}, 0',
        'rotate': '0, 0, -90',
        'direct_type': 'binaural_R'}

    roomsim = RoomSimulator(room_config=room_config, source_config=None,
                            receiver_config=receiver_config)

    azi = 0
    azi_rad = azi/180*np.pi
    source_config = configparser.ConfigParser()
    source_config['Source'] = {
        'pos': f'{2+1*np.cos(azi_rad)}, {2+np.sin(azi_rad)}, 2',
        'rotate': '0, 0, 0',
        'directivity': 'omnidirectional'}
    roomsim.load_source_config(source_config)

    roomsim.cal_all_img()
    rir_python = roomsim.cal_ir_mic()

    rir_matlab = sio.loadmat('H_ROOM_MIT_S1.mat', squeeze_me=True)['data']

    fig, ax = plt.subplots(1, 2, sharey=True, tight_layout=True,
                           figsize=(8, 3))
    ax[0].plot(rir_matlab[:, 0], label='roomsim')
    ax[0].plot(rir_python[:, 0], label='roomsimulator')
    ax[0].set_xlim([0, 500])
    ax[0].set_title('left ear')
    ax[0].legend()
    ax[1].plot(rir_matlab[:, 1])
    ax[1].plot(rir_python[:, 1])
    ax[1].set_title('right ear')
    fig.savefig('../images/validation.png', dpi=100)
