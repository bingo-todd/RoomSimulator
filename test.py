from BasicTools import wav_tools
import numpy as np


if __name__ == "__main__":
    src, fs = wav_tools.read_wav('speech.wav')
    rir = np.load('rir.npy')
    record = wav_tools.brir_filter(src, rir)
    wav_tools.write_wav(record, fs, 'record.wav')