import configparser
import numpy as np
import matplotlib.pyplot as plt

from RoomSimulator import RoomSimulator


room_config = configparser.ConfigParser()
room_config['Room'] = {
    'size': '4, 4, 4',
    'RT60': ', '.join([f'{item}' for item in np.ones(6) * 0.2]),
    'A': '',
    'Fs': 8000,
    'reflect_order': 10}

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

azi = 0
azi_rad = azi/180*np.pi
source_config = configparser.ConfigParser()
source_config['Source'] = {
    'pos': f'{2+1*np.cos(azi_rad)}, {2+np.sin(azi_rad)}, 2',
    'rotate': '0, 0, 0',
    'directivity': 'omnidirectional'}


def main():
    # no HP_filter
    roomsim = RoomSimulator(room_config=room_config,
                            source_config=source_config,
                            receiver_config=receiver_config)
    roomsim.cal_all_img()
    rir_no_HP = roomsim.cal_ir_mic()
    rir_len = rir_no_HP.shape[0]

    # with HP_filter
    room_config['Room'] = {**dict(room_config['Room']), 'HP_cutoff': 100}
    roomsim = RoomSimulator(room_config=room_config,
                            source_config=source_config,
                            receiver_config=receiver_config)
    roomsim._plot_HP_spec('../images/HP_filter_spectrum.png')
    roomsim.cal_all_img()
    rir_HP = roomsim.cal_ir_mic()

    fig, ax = plt.subplots(1, 2, tight_layout=True)
    ax[0].plot(rir_no_HP[:, 0])
    ax[0].plot(rir_HP[:, 0])
    ax[0].set_title('ir')

    ax[1].plot(np.arange(rir_len)/rir_len,
               np.log10(np.abs(np.fft.fft(rir_no_HP[:, 0]))+1e-20),
               label='no_HP')
    ax[1].plot(np.arange(rir_len)/rir_len,
               np.log10(np.abs(np.fft.fft(rir_HP[:, 0]))+1e-20),
               label='HP')
    ax[1].set_xlim([0, 100/rir_no_HP.shape[0]])
    ax[1].set_xlabel('norm_freq')
    ax[1].set_title('spectrum')
    fig.savefig('../images/rir_HP_effect.png')


if __name__ == '__main__':
    main()
