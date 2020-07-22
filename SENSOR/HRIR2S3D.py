import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def H3D2S3D():
    hrir_all = scipy.io.loadmat('Types/binaural.mat', squeeze_me=True)['H3D']
    # hrir_all=[azi_index, ele_index]
    # azi: 0 is the front, increase as the source move in clockwise direction

    ele_all = np.asarray([-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int)
    # azi number in each elevation
    n_azi_all = np.asarray([56, 60, 72, 72, 72, 72, 72, 60, 56, 45, 36, 24, 12, 1], dtype=np.int)

    S3D_L = np.empty((361, 181), object)
    S3D_R = np.empty((361, 181), object)
    for ele_i, ele in enumerate(ele_all):
        n_azi = n_azi_all[ele_i]
        azi_all = np.linspace(0, 360, n_azi+1)[:-1]
        for azi_i, azi in enumerate(azi_all):
            hrir = hrir_all[ele_i, azi_i]
            ele_index = np.int(np.round(ele + 90))

            # left is negative, right is positive
            azi_index_L = np.int(np.round(np.mod(azi+90+180, 360)))  
            # +90: listener's front is the 90 of left ear
            # +180: move the direct front in the middle
            S3D_L[azi_index_L, ele_index] = hrir[:, 0]

            azi_index_R = np.int(np.round(np.mod(azi-90+180, 360)))  
            # -90: listener's front is the -90 of right ear 
            # the front area of listener is the left of right ear, so - is needed
            # +180: move the direct front in the middle
            S3D_R[azi_index_R, ele_index] = hrir[:, 1]

    # -180 and +180 are the same direction
    S3D_L[360, :] = S3D_L[0, :]
    S3D_R[360, :] = S3D_R[0, :]

    np.save('Types/binaural_L.npy', S3D_L)
    np.save('Types/binaural_R.npy', S3D_R)

    fig, ax = plt.subplots(7, 2, sharex=True, sharey=True)
    for i, azi_i in enumerate(range(0, 361, 60)):
        ax[i, 0].plot(S3D_L[azi_i, 90])
        ax[i, 0].set_ylabel(f'{azi_i-180}')
        ax[i, 1].plot(S3D_R[azi_i, 90])
    ax[0, 0].set_title('L')
    ax[0, 1].set_title('R')
    fig.savefig('../img/directivity/HRIR2S3D.png')

if __name__ == '__main__':
    H3D2S3D()
