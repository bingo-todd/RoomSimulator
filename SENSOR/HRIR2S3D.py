import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def H3D2S3D():
    hrir_all = scipy.io.loadmat('Types/binaural.mat', squeeze_me=True)['H3D']

    ele_all = np.asarray([-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int)
    # azi number in each elevation
    n_azi_all = np.asarray([56, 60, 72, 72, 72, 72, 72, 60, 56, 45, 36, 24, 12, 1], dtype=np.int)

    S3D_L = np.empty((361, 181), object)
    S3D_R = np.empty((361, 181), object)
    for ele_i, ele in enumerate(ele_all):
        n_azi = n_azi_all[ele_i]
        azi_step = 360./n_azi
        azi_all = np.linspace(0, 360, n_azi+1)[:-1]
        for azi_i, azi in enumerate(azi_all):
            hrir = hrir_all[ele_i, azi_i]
            ele_index = np.int(ele + 90)
            azi_index = np.int(np.mod(azi+180, 360))
            S3D_L[azi_index, ele_index] = hrir[:, 0]
            S3D_R[azi_index, ele_index] = hrir[:, 1]
    np.save('Types/binaural_L.npy', S3D_L)
    np.save('Types/binaural_R.npy', S3D_R)


if __name__ == '__main__':
    H3D2S3D()
