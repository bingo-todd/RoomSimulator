from BasicTools import wav_tools
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


if __name__ == "__main__":
    src, fs = wav_tools.read_wav('speech.wav')
    rir = np.load('rir.npy')
    # record = wav_tools.brir_filter(src, rir)
    # wav_tools.write_wav(record, fs, 'record.wav')

    rir_ref = scipy.io.loadmat('pos_1_S1.mat', squeeze_me=True)['data']
    print(rir_ref)
    
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(rir[:, 0])
    ax[0, 1].plot(rir[:, 1])
    
    ax[1, 0].plot(rir_ref[:, 0])
    ax[1, 1].plot(rir_ref[:, 1])

    plt.show()
    fig.savefig('test.png')