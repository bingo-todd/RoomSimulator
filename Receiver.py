import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from AxisTransform import cal_transform_matrix
from Directivity import Directivity
from utils import cal_dist, pole2cartesian, cartesian2pole


class Mic(object):
    def __init__(self, config):
        """
        direct_type: directivity type of mic
        pos: position relative to the device it is mounted
        view: azimuth, elevation, roll angle shift relative to device it is mounted
        """

        self._load_config(config)
        self.directivity = Directivity()
        self.directivity.load(self.direct_type)
        self.tm = cal_transform_matrix(self.view)
        self.pos_room = None
        self.tm_room = None

    def _load_config(self, config):
        self.pos = np.asarray([np.float32(item) for item in config['pos'].split()])
        self.view = np.asarray([np.float32(item) for item in config['view'].split()])
        self.direct_type = config['direct_type']

    def get_ir(self, angle):
        """
        angle: angle of sound source relative to mic, possible range
            angle_azimuth: -180 to 180 in step of 1
            angle_elevation: -90 to 90 in step of 1
        """
        return self.directivity.get_ir(angle)


class Receiver(object):
    def __init__(self, config):
        self._load_config(config)

        self.tm = cal_transform_matrix(self.view)
        # combine transform matrix of receiver and mic
        # view of mic is relative to receiver
        for mic in self.mic_all:
            mic.pos_room = self.pos + mic.pos
            mic.tm_room = self.tm*mic.tm

    def _load_config(self, config):
        config_receiver = config['Receiver']
        self.pos = np.asarray([np.float32(item) for item in config_receiver['pos'].split()])
        self.view = np.asarray([np.float32(item) for item in config_receiver['view'].split()])
        self.n_mic = np.int32(config_receiver['n_mic'])

        # configure about microphone
        self.mic_all = [Mic(config[f'Mic_{mic_i}']) for mic_i in range(self.n_mic)]

    def show(self, ax=None, receiver_type='linear'):
        arrow_plot_settings = {
            'arrow_length_ratio': 0.2,
            'pivot': 'tail',
            'linewidth': 1}
        line_plot_settings = {
            'arrow_length_ratio': 0,
            'pivot': 'tail',
            'linewidth': 3,
            'color': 'black'}

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for mic_i, mic in enumerate(self.mic_all):
            ax.plot([mic.pos[0]], [mic.pos[1]], [mic.pos[2]], 'ro', markersize=12)
            ax.quiver(*(mic.pos+self.pos), *pole2cartesian(mic.view[:2]), **arrow_plot_settings)

        if receiver_type == 'linear':
            ax.quiver(*(self.mic_all[0].pos+self.pos), *(self.mic_all[-1].pos-self.mic_all[0].pos),
                      **line_plot_settings)
            # direction of receiver
            pos_receiver_center = (self.mic_all[0].pos + self.mic_all[-1].pos)/2
            ax.quiver(*pos_receiver_center, *pole2cartesian(self.view[:2]),
                      **arrow_plot_settings)

            receiver_len = cal_dist(self.mic_all[0].pos, self.mic_all[-1].pos)
            ax.set_xlim([self.mic_all[0].pos[0]-receiver_len/2, self.mic_all[-1].pos[0]+receiver_len/2])
            ax.set_ylim([self.mic_all[0].pos[1]-receiver_len/2, self.mic_all[-1].pos[1]+receiver_len/2])
            ax.set_zlim([self.mic_all[0].pos[2]-receiver_len/2, self.mic_all[-1].pos[2]+receiver_len/2])
        plt.show()


if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config['Receiver'] = {'pos': '0 0 0',
                          'view': '0 0 0',
                          'n_mic': '2'}
    config['Mic_0'] = {'pos': '0 -5 0',
                       'view': '0 0 0',
                       'direct_type': 'omnidirectional'}
    config['Mic_1'] = {'pos': '0 5 0',
                       'view': '0 0 0',
                       'direct_type': 'omnidirectional'}
    receiver = Receiver(config)
    receiver.show()
