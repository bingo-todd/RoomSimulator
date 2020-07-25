import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from BasicTools import get_fpath, nd_index, wav_tools
import logging


class Directivity(object):
    direct_dir = 'SENSOR/Types'
    direct_type_all = [
        'bidirectional', 'cardoid', 'dipole', 'hemisphere',
        'hypercardoid', 'null_sensor', 'omnidirectional',
        'subcardoid', 'supercardoid', 'unidirectional',
        'binaural_L', 'binaural_R']
    ele_range = [-90, 90]
    azi_range = [-180, 180]

    def __init__(self, Fs):
        self.Fs = Fs
        self.valid_index_all = None

    def load(self, direct_type):
        """ S3D: sensitivity of mic in 3D
        direct_type: directivity types,
        supported type: bidirectional, cardoid, dipole, hemisphere,
                        hypercardoid, null_sensor, omnidirectional,
                        subercardoid, supercardoid, unidirectional,
                        binaural_L, binaural_R
        """
        if direct_type not in self.direct_type_all:
            print(f'supported directivity type: {direct_type}')
            raise Exception(f'unknown type of directivity type {direct_type}')

        self.direct_type = direct_type
        self.S3D = np.load(f'{self.direct_dir}/{direct_type}.npy', allow_pickle=True)
        valid_index_all = [[i, j] 
                            for i in range(self.S3D.shape[0]) 
                            for j in range(self.S3D.shape[1]) 
                            if self.S3D[i, j] is not None]
        self.valid_index_all = np.asarray(valid_index_all) 

        # MIT HRIR Fs=44100, if input Fs is not 44100, resample S3D
        if self.Fs is not None and self.Fs != 44100 and self.S3D.dtype == object:
            logging.warning(f'resample hrir from 44100 to {self.Fs}')
            print('resample')
            for i, j in self.valid_index_all:
                self.S3D[i, j] = wav_tools.resample(self.S3D[i, j], 44100, self.Fs)

    @classmethod
    def angle2index(self, angle):
        azi_index = np.int(np.round(np.mod(angle[0]+180, 360)))
        ele_index = np.int(np.round(angle[1]+90))
        return azi_index, ele_index

    def get_ir(self, angle):
        azi_i, ele_i = self.angle2index(angle)
        ir = self.S3D[azi_i, ele_i]
        if ir is None:
            if self.direct_type == 'binaural_L' or self.direct_type == 'binaural_R':
                if angle[1] <- 40:  # elevation = -40 ~ 90 
                    return None
            dist_all = np.sum((self.valid_index_all-np.asarray([[azi_i, ele_i]]))**2, axis=1)
            azi_i, ele_i = self.valid_index_all[np.argmin(dist_all)]
        return self.S3D[azi_i, ele_i]

    @classmethod
    def validate(self, is_plot=True):
        for direct_type in self.direct_type_all:
            file_path = f'{self.direct_dir}/{direct_type}.npy'
            if not os.path.exists(file_path):
                logging.warning(f'{direct_type} do not exists')
            if is_plot:
                S3D = np.load(file_path, allow_pickle=True)
                if direct_type == 'binaural_L' or direct_type == 'binaural_R':
                    fig, ax = plt.subplots(5, 7, figsize=[12, 8], tight_layout=True, sharex=True, sharey=True)
                    for ele_i, ele in enumerate(range(-60, 61, 30)):
                        for azi_i, azi in enumerate(range(-180, 181, 60)):
                            ele_index = np.int(ele+90)
                            azi_index = np.int(azi+180)
                            ir = S3D[azi_index, ele_index]
                            if ir is None:
                                continue
                            ax[ele_i, azi_i].plot(ir, linewidth=2)
                            ax[0, azi_i].set_title(f'{azi}')
                        ax[ele_i, 0].set_ylabel(f'{ele}')
                else:
                    fig, ax = plt.subplots(1, 1)
                    ax.imshow(S3D, vmin=0, vmax=1, cmap='jet')
                os.makedirs('img/directivity', exist_ok=True)
                fig.savefig(f'img/directivity/{direct_type}.png')
                plt.close(fig)

    @classmethod
    def loadS3Dmat(self, file_path):
        data_mat = scipy.io.loadmat(file_path, squeeze_me=True)
        S3D = np.asarray(data_mat['S3D'], dtype=np.float32).T
        return S3D

    @classmethod
    def mat2npy(self):
        for direct_type in self.direct_type_all:
            mat_path = f'{self.direct_dir}/{direct_type}.mat'
            S3D = self.loadS3Dmat(mat_path)
            npy_path = mat_path.replace('mat', 'npy')
            np.save(npy_path, S3D)
            print(npy_path)


if __name__ == "__main__":
    # Directivity.mat2npy()
    Directivity.validate(is_plot=True)