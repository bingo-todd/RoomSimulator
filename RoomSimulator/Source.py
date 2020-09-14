import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

from .AxisTransform import view2tm
from .Directivity import Directivity


class Source(object):
    def __init__(self, config):
        """
        pos: position of source in room
        view: azimuth, elevation, roll angle of sound source
        direct_type: directivity type of source
        """
        self._load_config(config)
        
        self.directivity = Directivity(self.Fs)
        self.directivity.load(self.direct_type)
        self.tm = view2tm(self.view)

    def _load_config(self, config):
        self.Fs = np.float(config['Fs'])
        self.pos = np.asarray([np.float32(item) for item in config['pos'].split(',')])
        self.view = np.asarray([np.float32(item) for item in config['view'].split(',')])
        self.direct_type = config['directivity']

    @staticmethod
    def angle2index(self, angle):
        azi_index = np.int16(np.mod(angle[0] + 180, 360))
        ele_index = np.int16(np.mod(angle[1] + 90, 180))
        return azi_index, ele_index

    def get_ir(self, angle):
        """
        angle: angle of sound source relative to mic, possible range
            angle_azimuth: -180 to 180 in step of 1
            angle_elevation: -90 to 90 in step of 1
        """
        return self.directivity.get_ir(angle)

    def show(self, ax):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.pos, marker='+')
        plt.show()


if __name__ == '__main__':
    import configparser
    config = configparser.ConfigParser()
    config['Source'] = {'pos': '0 0 0',
                        'view': '0 0 0',
                        'direct_type': 'omnidirectional'}
    source = Source(config['Source'])
    source.show()
