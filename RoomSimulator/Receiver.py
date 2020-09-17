import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

from .AxisTransform import rotate2tm
from .Directivity import Directivity
from .utils import cal_dist, pole2cartesian, cartesian2pole


class Mic(object):
    def __init__(self, config):
        """
        direct_type: directivity type of mic
        pos: position relative to the device it is mounted
        rotate: azimuth, elevation, roll angle shift relative to device it is mounted
        """

        self._load_config(config)

        self.directivity = Directivity(self.Fs)
        self.directivity.load(self.direct_type)
        self.tm = rotate2tm(-self.rotate)
        self.pos_room = None
        self.rotate_room = None
        self.tm_room = None

    def _load_config(self, config):
        self.Fs = np.float(config['Fs'])
        self.pos = np.asarray([np.float32(item) for item in config['pos'].split(',')])
        self.rotate = np.asarray([np.float32(item) for item in config['rotate'].split(',')])
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

        self.tm = rotate2tm(-self.rotate)

        self.direct_type = 'omnidirectional'  # 
        self.directivity = Directivity(self.Fs)
        self.directivity.load(self.direct_type)
        # combine transform matrix of receiver and mic
        # rotate of mic is relative to receiver
        for mic in self.mic_all:
            mic.pos_room = self.pos + mic.pos
            mic.tm_room = np.matmul(mic.tm, self.tm) 
            mic.rotate_room = self.rotate + mic.rotate

    def _load_config(self, config):
        config_receiver = config['Receiver']
        self.Fs = np.float(config_receiver['Fs'])
        self.pos = np.asarray([np.float32(item) for item in config_receiver['pos'].split(',')])
        self.rotate = np.asarray([np.float32(item) for item in config_receiver['rotate'].split(',')])
        self.n_mic = np.int32(config_receiver['n_mic'])

        # configure about microphone
        self.mic_all = []
        for mic_i in range(self.n_mic):
            config[f'Mic_{mic_i}']['Fs'] = f'{self.Fs}'
            self.mic_all.append(Mic(config[f'Mic_{mic_i}']))

    def show(self, ax=None, receiver_type='linear', arrow_len=1):
        x_arrow_plot_settings = {
            'arrow_length_ratio': 0.4,
            'pivot': 'tail',
            'color': [1, 20./255, 147./255],
            'linewidth': 2}
        y_arrow_plot_settings = {
            'arrow_length_ratio': 0.4,
            'pivot': 'tail',
            'color': [155./255, 48./255, 1],
            'linewidth': 2}
        z_arrow_plot_settings = {
            'arrow_length_ratio': 0.4,
            'pivot': 'tail',
            'color': [1, 165./255, 0],
            'linewidth': 2}
        line_plot_settings = {
            'arrow_length_ratio': 0,
            'pivot': 'tail',
            'linewidth': 2,
            'color': 'black'}

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        for mic_i, mic in enumerate(self.mic_all):
            if mic_i == 0:
                label = 'Mic'
            else:
                label = None
            ax.plot([mic.pos_room[0]], [mic.pos_room[1]], [mic.pos_room[2]], 'ko', markersize=8, 
                     alpha=0.5, label=label)
            mic_tm_room = np.matmul(rotate2tm(mic.rotate), self.tm) 
            ax.quiver(*mic.pos_room, *mic_tm_room[:, 0]*arrow_len, **x_arrow_plot_settings)
            ax.quiver(*mic.pos_room, *mic_tm_room[:, 1]*arrow_len, **y_arrow_plot_settings)
            ax.quiver(*mic.pos_room, *mic_tm_room[:, 2]*arrow_len, **z_arrow_plot_settings)

        if receiver_type == 'linear':
            ax.quiver(*(self.mic_all[0].pos+self.pos), *(self.mic_all[-1].pos-self.mic_all[0].pos), alpha=0.2,
                      **line_plot_settings)
            # direction of receiver
            receiver_tm = rotate2tm(self.rotate)
            pos_receiver_center = (self.mic_all[0].pos_room + self.mic_all[-1].pos_room)/2
            ax.quiver(*pos_receiver_center, *receiver_tm[:,0]*arrow_len, label='x_direct', **x_arrow_plot_settings)
            ax.quiver(*pos_receiver_center, *receiver_tm[:,1]*arrow_len, label='y_direct', **y_arrow_plot_settings)
            ax.quiver(*pos_receiver_center, *receiver_tm[:,2]*arrow_len, label='z_direct', **z_arrow_plot_settings)
            
    plt.show()


if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config['Receiver'] = {'Fs': '44100',
                          'pos': '0, 0, 0',
                          'rotate': '0, 0, 0',
                          'n_mic': '2'}
    config['Mic_0'] = {'pos': '0, 5, 0',
                       'rotate': '-90, 0, 0',
                       'direct_type': 'binaural_L'}
    config['Mic_1'] = {'pos': '0, -5, 0',
                       'rotate': '90, 0, 0',
                       'direct_type': 'binaural_R'}
    receiver = Receiver(config)

    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True)
    for i in range(4):
        rir = receiver.mic_all[0].get_ir([i*20, 0])
        ax[i].plot(rir)
    plt.show()

    print(receiver.tm, '\n', receiver.mic_all[0].tm)

    print(np.matmul(receiver.mic_all[0].tm.T, np.asarray([1, 0, 0])))
    raise Exception()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    receiver.show(ax)
    ax.legend()
    ax.set_xlim([-6, 6]); ax.set_xlabel('x')
    ax.set_ylim([-6, 6]); ax.set_ylabel('y')
    ax.set_zlim([-6, 6]); ax.set_zlabel('z')

    os.makedirs('img/Receiver', exist_ok=True)
    fig.savefig('img/Receiver/demo.png')
