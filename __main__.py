import configparser 
import numpy as np
import sys
import matplotlib.pyplot as plt

from .RoomSimulator.RoomSimulator import RoomSimulator


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
        'view': '0, 0, 0',
        'n_mic': '2'}
    head_r = 0.145/2
    receiver_config['Mic_0'] = {
        'pos': f'0, {head_r}, 0',
        'view': '0, 0, 90',
        'direct_type': 'binaural_L'}
    receiver_config['Mic_1'] = {
        'pos': f'0, {-head_r}, 0',
        'view': '0, 0, -90',
        'direct_type': 'binaural_R'}

    roomsim = RoomSimulator(room_config=room_config, source_config=None, receiver_config=receiver_config)

    azi = 0
    azi_rad = azi/180*np.pi
    source_config = configparser.ConfigParser()
    source_config['Source'] = {
        'pos': f'{2+1*np.cos(azi_rad)}, {2+np.sin(azi_rad)}, 2',
        'view': '0, 0, 0',
        'directivity':'omnidirectional'}
    roomsim.load_source_config(source_config)
    fig, ax = roomsim.show()
    ax.view_init(elev=60, azim=30)
    os.makedirs('images')
    fig.savefig(f'images/room.png', dpi=200)
    plt.close(fig)
         
    fig, ax = roomsim.cal_all_img(is_plot=True)
    fig.savefig('images/image_sources.png')

    rir = roomsim.cal_ir_mic(is_verbose=False, img_dir='img/verbose')
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(rir[:, 0])
    ax[1].plot(rir[:, 1])
    ax[2].plot(rir[:, 0] - rir[:, 1])
    fig.savefig('images/rir.png')

    np.save('rir.npy', rir)
