import configparser
import numpy as np
import time
import matplotlib.pyplot as plt

from RoomSimulator import RoomSimulator


if __name__ == '__main__':

    room_config = configparser.ConfigParser()
    room_config['Room'] = {
        'size': '4, 4, 4',
        'RT60': ', '.join([f'{item}' for item in np.ones(6) * 0.2]),
        'A': '',
        'Fs': 8000,
        'reflect_order': 10,
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

    roomsim._plot_HP_spec('../images/HP_filter_spectrum.png')

    azi = 0
    azi_rad = azi/180*np.pi
    source_config = configparser.ConfigParser()
    source_config['Source'] = {
        'pos': f'{2+1*np.cos(azi_rad)}, {2+np.sin(azi_rad)}, 2',
        'rotate': '0, 0, 0',
        'directivity': 'omnidirectional'}
    roomsim.load_source_config(source_config)

    t_start = time.time()
    roomsim.cal_all_img_direct()
    rir_raw = roomsim.cal_ir_mic()
    t_end = time.time()
    print(f'raw_version elapsed time: {t_end-t_start}s')

    t_start = time.time()
    roomsim.cal_all_img()
    rir_compact = roomsim.cal_ir_mic()
    t_end = time.time()
    print(f'compact_version elapsed time: {t_end-t_start}s')

    fig, ax = plt.subplots(1, 3)
    ax[0].plot(rir_raw[:, 0])
    ax[1].plot(rir_compact[:, 0])
    ax[2].plot(rir_raw[:, 0] - rir_compact[:, 0])
    fig.savefig('../images/rir_raw_compact.png')

    # np.save('rir.npy', rir)
