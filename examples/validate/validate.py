import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from BasicTools import wav_tools

import sys
sys.path.append('../')
from syn_inter_BRIR import parse_config_file  #noqa: E402
from syn_tar_BRIR import syn_brir  #noqa: E402


def main(config_path):

    src, fs = wav_tools.read_wav('data/src.wav')

    new_brir_path = 'data/test.wav'
    # configs = parse_config_file(config_path)
    # new_config_path = 'config.cfg'
    # new_fig_path = 'test.png'
    # syn_brir(configs, new_config_path, new_brir_path, new_fig_path,
    #      parallel_type=2, n_worker=12)
    new_brir, fs = wav_tools.read_wav(new_brir_path)
    new_record = wav_tools.brir_filter(src, new_brir)
    wav_tools.write_wav(new_record, fs, 'data/new/reverb/15_0387_0.wav')

    brir, fs = wav_tools.read_wav(config_path.replace('cfg', 'wav'))
    record = wav_tools.brir_filter(src, brir)
    wav_tools.write_wav(new_record, fs, 'data/pre/reverb/15_0387_0.wav')

    # brir
    fig, ax = plt.subplots(3, 3, tight_layout=True, figsize=[10, 8])
    ax[0, 0].plot(brir[:, 0])
    ax[0, 0].set_ylabel('brir')
    ax[0, 0].set_title('pre')
    ax[0, 1].plot(new_brir[:, 0])
    ax[0, 1].set_title('new')
    ax[0, 2].plot(brir[:, 0] - new_brir[:, 0])
    ax[0, 2].yaxis.set_major_formatter(ticker.LogFormatter())
    ax[0, 2].set_title('difference')

    ax[1, 0].plot(record[:, 0])
    ax[1, 0].set_ylabel('record')
    ax[1, 1].plot(new_record[:, 0])
    ax[1, 2].plot(record[:, 0] - new_record[:, 0])
    ax[1, 2].yaxis.set_major_formatter(ticker.LogFormatter())

    specgram, freqs, bins, im = ax[2, 0].specgram(record[:, 0], Fs=fs,
                                                  NFFT=512, noverlap=256,
                                                  cmap='jet')
    new_specgram, freqs, bins, im = ax[2, 1].specgram(new_record[:, 0], Fs=fs,
                                                      NFFT=512, noverlap=256,
                                                      cmap='jet')
    ax[2, 2].imshow(specgram-new_specgram, aspect='auto', cmap='jet',
                    extent=[bins[0], bins[-1], freqs[0], freqs[-1]])
    fig.savefig('images/validate.png')


if __name__ == '__main__':
    import sys
    config_path = sys.argv[1]
    main(config_path)
