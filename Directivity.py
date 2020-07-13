import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from BasicTools import get_fpath, nd_index
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

    def __init__(self):
        None

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
        self.S3D = np.load(f'{self.direct_dir}/{direct_type}.npy', allow_pickle=True)
        index_0, index_1 = np.nonzero(self.S3D is not None)
        self.valid_index = np.concatenate((index_0[:, np.newaxis], index_1[:, np.newaxis]), axis=1)

    @classmethod
    def angle2index(self, angle):
        azi_index = np.int16(np.mod(angle[0]+180, 360))
        ele_index = np.int16(angle[1]+90)
        return azi_index, ele_index

    def get_ir(self, angle):
        azi_i, ele_i = self.angle2index(angle)
        ir = self.S3D[azi_i, ele_i]
        if ir is None:
            dist_all = np.sum((self.valid_index-np.asarray([[azi_i, ele_i]]))**2)
            azi_i_valid, ele_i_valid = self.valid_index[np.argmax()]
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
                    ele = 0
                    fig, ax = plt.subplots(4, 2, figsize=[10, 12], tight_layout=True)
                    for i in range(8):
                        azi = 45*i-180
                        azi_index, ele_index = self.angle2index([azi, ele])
                        ir = S3D[azi_index, ele_index]
                        ax_i, ax_j = nd_index(i, [4, 2])
                        ax[ax_i, ax_j].plot(ir, linewidth=2)
                        ax[ax_i, ax_j].set_title(f'azi:{azi}')
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

    # direct = Directivity('omnidirectional')
    # direct.validate(is_plot=True)