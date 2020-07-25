from BasicTools import wav_tools
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def test_rir():
    src, fs = wav_tools.read_wav('speech.wav')
    rir = np.load('rir.npy')
    # record = wav_tools.brir_filter(src, rir)
    # wav_tools.write_wav(record, fs, 'record.wav')

    rir_ref = scipy.io.loadmat('pos_1_S1.mat', squeeze_me=True)['data']
    
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].plot(rir[:, 0])
    ax[0, 1].plot(rir[:, 1]-rir[:, 0])
    ax[0, 2].plot(rir[:, 1])
    ax[0, 2].plot(rir[:, 0])
    print(np.max(np.abs(rir[:,0]-rir[:,1])))
    
    ax[1, 0].plot(rir_ref[:, 0])
    ax[1, 1].plot(rir_ref[:, 1])

    plt.show()
    fig.savefig('test.png')


def test_fft():
    fs = 16000
    spec_amp1 = np.ones(256)
    spec_amp2 = np.ones(512)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.real(np.fft.ifft(spec_amp1)))
    ax[1].plot(np.real(np.fft.ifft(spec_amp2)))
    plt.show()


def test_hrir():
    hrir_L_path = 'SENSOR/Types/binaural_L.npy'
    hrir_L = np.load(hrir_L_path, allow_pickle=True)

    hrir_R_path = 'SENSOR/Types/binaural_R.npy'
    hrir_R = np.load(hrir_R_path, allow_pickle=True)
    
    for i in range(361):
        if hrir_L[i, 90] is None and hrir_R[360-i, 90] is None:
            continue
        fig, ax = plt.subplots(1, 1)
        ax.plot(hrir_L[i, 90])
        ax.plot(hrir_R[360-i, 90])
        ax.set_title(f'{i}')
        fig.savefig(f'brir_comp_{i}.png')
        plt.close(fig)


if __name__ == '__main__':
    test_hrir()

