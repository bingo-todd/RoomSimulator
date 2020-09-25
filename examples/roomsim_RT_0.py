import configparser
import numpy as np

from RoomSimulator import RoomSimulator


if __name__ == '__main__':

    room_config = configparser.ConfigParser()
    room_config['Room'] = {
        'size': '4, 4, 4',
        'RT60': ', '.join([f'{item}' for item in np.ones(6) * 0]),
        'A': '',
        'Fs': 16000,
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

    roomsim._plot_HP_spec('../images/HP_filter_spectrum.png')

    azi = 0
    azi_rad = azi/180*np.pi
    source_config = configparser.ConfigParser()
    source_config['Source'] = {
        'pos': f'{2+1*np.cos(azi_rad)}, {2+np.sin(azi_rad)}, 2',
        'rotate': '0, 0, 0',
        'directivity': 'omnidirectional'}
    roomsim.load_source_config(source_config)

    roomsim.cal_all_img()
    rir = roomsim.cal_ir_mic()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.plot(rir)
    fig.savefig('../images/rir_RT_0.png')
