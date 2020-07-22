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


if __name__ == '__main__':
    test_rir()