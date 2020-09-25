import configparser
import numpy as np
import matplotlib.pyplot as plt

from RoomSimulator import RoomSimulator


if __name__ == '__main__':

    room_config = configparser.ConfigParser()
    room_config['Room'] = {
        'size': '12, 6, 4',
        'RT60': ', '.join([f'{item}' for item in np.ones(6) * 0.2]),
        'A': '',
        'Fs': 16000,
        'reflect_order': -1,
        'HP_cutoff': 100}

    receiver_config = configparser.ConfigParser()
    receiver_config['Receiver'] = {
        'pos': '1, 3, 2',
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

    fig, ax = plt.subplots(2, 3, figsize=(10, 6), tight_layout=True)
    ax = ax.flatten()
    dist_all = [1, 2, 4, 6, 8]
    brirs_all = []
    for dist_i, dist in enumerate(dist_all):
        source_config = configparser.ConfigParser()
        source_config['Source'] = {
            'pos': f'{1+dist}, 3, 2',
            'rotate': '0, 0, 0',
            'directivity': 'omnidirectional'}
        roomsim.load_source_config(source_config)
        roomsim.cal_all_img()
        rir = roomsim.cal_ir_mic()
        ax[dist_i].plot(rir)
        ax_spec = ax[dist_i].twinx()
        ax_spec.plot(np.log10(np.abs(np.fft.fft(rir[:, 0]))+1e-20))
        ax[dist_i].set_title(f'{dist}')
    fig.savefig('../images/rir_diff_dist.png')
