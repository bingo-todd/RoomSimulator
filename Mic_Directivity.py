import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

data_dir = 'SENSOR/Types/'


def load_S3D(direct_type):
    """ S3D sensitivity of mic in 3D
    direct_type: directivity types,
    supported type: bidirectional, cardoid, dipole, hemisphere,
                    hypercardoid, null_sensor, omnidirectional,
                    subercardoid, supercardoid, unidirectional
    """
    direct_type_all = [
        'bidirectional', 'cardoid', 'dipole', 'hemisphere',
        'hypercardoid', 'null_sensor', 'omnidirectional',
        'subercardoid', 'supercardoid', 'unidirectional'
    ]
    if direct_type not in direct_type_all:
        print(f'supported directivity type: {direct_type_all}')
        raise Exception('unknown type of directivity table')

    data_mat = sio.loadmat(f'{data_dir}/{direct_type}.mat', squeeze_me=True)
    S3D = np.asarray(data_mat['S3D'], dtype=np.float32)
    return S3D


def validate(is_plot=True):
    fname_all = os.listdir(data_dir)
    for fname in fname_all:
        direct_type, ext = fname.split('.')
        if ext != 'mat':
            continue
        S3D = load_S3D(direct_type)

        if is_plot:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(S3D, vmin=0, vmax=1, cmap='jet')
            fig.savefig(f'{data_dir}/{fname_no_ext}.png')

if __name__ == '__main__':
    validate()
