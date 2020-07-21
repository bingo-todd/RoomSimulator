import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import configparser
import matplotlib.pyplot as plt
import os
import logging

from AxisTransform import view2tm
from DelayFilter import DelayFilter
from Reverb import RT2Absorb
from Room import ShoeBox
from Source import Source
from Receiver import Receiver
from utils import cal_dist, cartesian2pole, pole2cartesian, filter
from BasicTools import ProcessBar


class RoomSim(object):
    def __init__(self, config):

        self._load_config(config)

        self.T_Fs = 1. / self.Fs
        self.F_nyquist = self.Fs/2.0  # Half sampling frequency

        # constants
        self.F_abs = np.array([125, 250, 500, 1000, 2000, 4000])
        self.F_abs_norm = self.F_abs/self.F_nyquist
        self.F_abs_extend = np.concatenate(([0], self.F_abs, [self.F_nyquist]))
        self.F_abs_extend_norm = self.F_abs_extend/self.F_nyquist
        self.Two_pi = 2*np.pi

        self.c_air = 343.0  # sound speed in air
        self.Fs_c = self.Fs/self.c_air  # Samples per metre

        # high pass filter to remove DC
        self._HP_filter_coef = self._make_HP_filter()

        # calculate ir single reflection based on spectrum amplitude
        n_fft = 512 # np.int32(2*np.round(self.F_nyquist/self.F_abs[1]))
        if np.mod(n_fft, 2) == 1:
            n_fft = n_fft + 1
        self.n_fft = n_fft
        self.n_fft_half_valid = np.int32(self.n_fft/2) # dc component is not considered
        self.freq_norm_all = np.arange(self.n_fft_half_valid+1)/self.n_fft_half_valid
        self.window = np.hanning(self.n_fft)
        self.zero_padd_array = np.zeros(n_fft)

        RT60 = self.room.RT60
        self.ir_len = np.int(np.floor(np.max(RT60)*self.Fs))

        self.n_cube_xyz = np.ceil(self.ir_len/self.Fs_c/(self.room.size*2))
        self.n_img_in_cube = 8

        # init
        self.amp_gain_reflect_all = np.zeros((0, self.F_abs.shape[0]))

        m_air = 6.875e-4*(self.F_abs/1000)**1.7
        self.air_attenuate_per_dist = np.exp(-0.5*m_air).T

        # threshold of ir amplitude
        self.reflect_amp_theta = 1e-5

        self.img_pos_all = np.zeros((0, 3))
        self.reflect_attenuate_all = np.zeros((0, self.F_abs.shape[0]))
        self.n_img = 0

    def _load_config(self, config):
        # basic configuration
        config_basic = config['Basic']
        self.Fs = np.int32(config_basic['Fs'])
        self.HP_cutoff = np.float(config_basic['HP_cutoff'])
        self.reflect_order = np.int32(config_basic['reflect_order'])
        # configure about room
        self.room = ShoeBox(config['Room'])
        # configure about sound source
        # add Fs to source config
        config['Source']['Fs'] = f'{self.Fs}'
        self.source = Source(config['Source'])
        # configure about receiver
        # add Fs to receiver config
        config['Receiver']['Fs'] = f'{self.Fs}'
        config_receiver = configparser.ConfigParser()
        config_receiver['Receiver'] = config['Receiver']
        n_mic = np.int8(config['Receiver']['n_mic'])
        # configure about microphone
        for mic_i in range(n_mic):
            mic_key = f'Mic_{mic_i}'
            config_receiver[mic_key] = config[mic_key]
        self.receiver = Receiver(config_receiver)

    def show(self):
        fig, ax = self.room.show()
        ax.scatter(*self.source.pos, 'ro')
        self.receiver.show(ax)
        ax.legend()
        return fig, ax

    def _make_HP_filter(self):
        # Second order high-pass IIR filter to remove DC buildup
        # (nominal -4dB cut-off at 20 Hz)
        # cutoff frequency 
        if self.HP_cutoff is None:
            return None
        w = 2*np.pi*self.HP_cutoff
        r1, r2 = np.exp(-w*self.T_Fs), np.exp(-w*self.T_Fs)
        b1, b2 = -(1+r2), r2  # Numerator coefficients (fix zeros)
        a1, a2 = 2*r1*np.cos(w*self.T_Fs), -r1**2  # Denominator coefficients
        HP_gain = (1-b1+b2)/(1+a1-a2)  # Normalisation gain
        b = np.asarray([1, b1, b2])/HP_gain
        a = np.asarray([1, -a1, -a2])
        return b, a

    def HP_filter(self, x):
        if self._HP_filter_coef is None:
            return x
        else:
            return filter(*self._HP_filter_coef, x)

    def cal_b_power(self, wall_i, n_reflect):
        """i: index of B
        n: power term
        """
        # TODO: 可以将指数分解，尽量根据已有的数据计算幂指数
        key = f'{wall_i}_{n_reflect}'
        if key not in self.B_power_table.keys():
            self.B_power_table[key] = self.room.B[wall_i]**n_reflect
        return self.B_power_table[key]

    def cal_B_power(self, n_all):
        result = np.ones(self.F_abs.shape[0])
        for wall_i in range(6):
            result = result * self.cal_b_power(wall_i, n_all[wall_i])
        return result

    def get_img(self, is_plot=False, is_verbose=False):
        # TODO input argments validate
        # use dictionary to save the exponent of B
        # init
        self.B_power_table = dict()

        # codes the eight permutations of x+/-xp, y+/-yp, z+/-zp
        # where [-1 -1 -1] identifies the parent source.
        img_pos_rel_in_cube = np.array(
            [[+1, +1, +1],
             [+1, -1, +1],
             [-1, -1, +1],
             [-1, +1, +1],
             # above: upper 4 images in cube
             # below: lower 4 images in cube
             [+1, +1, -1],
             [+1, -1, -1],
             [-1, -1, -1],
             [-1, +1, -1]])
        img_pos_in_cube = img_pos_rel_in_cube*self.source.pos[np.newaxis, :]

        # Includes/excludes bx, by, bz depending on 0/1 state.
        n_reflect_relative = np.array(
            [[+0, +0, +0],
             [+1, +0, +0],
             [+1, +1, +0],
             [+0, +1, +0],
             # above: upper 4 images in cube
             # below: lower 4 images in cube
             [+0, +0, +1],
             [+1, +0, +1],
             [+1, +1, +1],
             [+0, +1, +1]])

        n_F_abs = len(self.F_abs)

        # Maximum number of image sources
        n_img_max = np.int64(np.prod(2*self.n_cube_xyz+1)*8)
        img_pos_all = np.zeros((n_img_max, 3))  # image source co-ordinates
        reflect_attenuate_all = np.zeros((n_img_max, n_F_abs))
        n_img = -1  # number of significant images of each parent source

        logger = logging.getLogger()
        file_handler = logging.FileHandler('log/get_img.log', mode='w')
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        for cube_i_x in np.arange(-self.n_cube_xyz[0], self.n_cube_xyz[0]+1):
            cube_pos_x = cube_i_x * self.room.size_double[0]
            for cube_i_y in np.arange(-self.n_cube_xyz[1], self.n_cube_xyz[1]+1):
                cube_pos_y = cube_i_y * self.room.size_double[1]
                for cube_i_z in np.arange(-self.n_cube_xyz[2], self.n_cube_xyz[2]+1):
                    cube_pos_z = cube_i_z * self.room.size_double[2]
                    cube_pos = np.asarray([cube_pos_x, cube_pos_y, cube_pos_z])
                    cube_i_xyz = np.asarray([cube_i_x, cube_i_y, cube_i_z])
                    n_reflect_xyz1 = np.abs(cube_i_xyz).reshape([3, 1])
                    for i in np.arange(self.n_img_in_cube):
                        img_pos = cube_pos + img_pos_in_cube[i]
                        n_img = n_img+1
                        img_pos_all[n_img] = img_pos
                        n_reflect_xyz0 = n_reflect_xyz1 + n_reflect_relative[i].reshape([3, 1])
                        n_reflect_wall_all = np.reshape(
                            np.concatenate((n_reflect_xyz0, n_reflect_xyz1),
                                           axis=1),
                            [6, 1])
                        reflect_attenuate_all[n_img] = self.cal_B_power(np.squeeze(n_reflect_wall_all))

                        logger.info(f'{n_img}  {n_reflect_wall_all.reshape([-1])}  {img_pos_all[n_img]}')

        # Complete impulse response for the source
        n_img = n_img + 1
        self.img_pos_all = img_pos_all[:n_img]
        self.reflect_attenuate_all = reflect_attenuate_all
        self.n_img = n_img

        if is_plot:
            self.room.show(extra_point=self.img_pos_all)
            plt.title(f'{self.n_img}')
            plt.show()

    def amp_spec_to_ir(self, amp_reflect):
        # interpolated grid points
        amp_reflect_extend = np.concatenate((amp_reflect[:1], amp_reflect, amp_reflect[-1:]))
        spec_amp_half = interp1d(self.F_abs_extend_norm, amp_reflect_extend)(self.freq_norm_all)
        spec_amp = np.concatenate((spec_amp_half, np.conj(np.flip(spec_amp_half[1:-1]))))
        ir = np.real(np.fft.ifft(spec_amp))
        ir = self.window * np.concatenate((ir[self.n_fft_half_valid+1:self.n_fft],
                                           ir[:self.n_fft_half_valid+1]))
        return ir

    def _cal_ir_1mic(self, mic, logger):
        ir = np.zeros(self.ir_len)
        # pb = ProcessBar(self.n_img)
        for img_i in np.arange(self.n_img):
            # pb.update()
            if mic.direct_type == 'binaural_L' or mic.direct_type == 'binaural_R':
                pos_img_to_mic = np.matmul(mic.tm_room.T, (self.img_pos_all[img_i] - self.receiver.pos))
            else:
                pos_img_to_mic = np.matmul(mic.tm_room.T, (self.img_pos_all[img_i] - mic.pos_room))
            *angle_img_to_mic, dist = cartesian2pole(pos_img_to_mic)
            
            # 
            logger.info(f'{img_i}  {dist:.2f}  {angle_img_to_mic}')

            pos_mic_to_img = np.matmul(self.source.tm.T, (mic.pos_room - self.img_pos_all[img_i]))
            *angle_mic_to_img, _ = cartesian2pole(pos_mic_to_img)

            reflect_amp = self.reflect_attenuate_all[img_i]
            # energy loss because of distance
            reflect_amp = reflect_amp/dist
            # absorption due to air

            reflect_amp = reflect_amp*(self.air_attenuate_per_dist**dist)

            # calculate ir based on amp
            # 计算得到的ir_tmp对应的时间范围是：-n_fft/2:n_fft/2, 相当于已经延迟了n_fft/2，即n_fft/2
            ir_tmp = self.amp_spec_to_ir(reflect_amp)

            # directivity of sound source, directivity after imaged
            ir_source = self.source.get_ir(angle_mic_to_img)
            ir_tmp = filter(ir_source, 1, ir_tmp)

            # For primary sources, and image sources with impulse response
            # peak magnitudes >= -100dB (1/100000)
            if True:  # np.max(np.abs(ir_tmp)) >= self.reflect_amp_theta:
                # mic directivity filter
                ir_mic = mic.get_ir(angle_img_to_mic)
                if ir_mic is None:
                    continue
                # ir_tmp = filter(ir_mic, 1, np.concatenate((ir_tmp, self.zero_padd_array)))
                
                # parse delay into integer and fraction.
                delay_sample_num = dist * self.Fs_c
                delay_sample_num_int = np.int32(np.round(delay_sample_num))
                delay_sample_num_frac = delay_sample_num - delay_sample_num_int

                # apply fraction delay to ir_tmp
                ir_tmp = DelayFilter(self.Fs, delay_sample_num_frac/self.Fs).filter(ir_tmp, is_padd=True)                 

                # apply integer delay
                # first shift ir_tmp which has delay capacity of n_fft_half_valid
                start_index_0 = max([self.n_fft_half_valid-delay_sample_num_int, 0])
                ir_tmp = ir_tmp[start_index_0:]
                
                # if delay_sample_num_int is larger than n_fft_half_valid
                # apply the remain delay while add ir_tmp to ir_all
                start_index_1 = max([delay_sample_num_int-self.n_fft_half_valid, 0])
                if start_index_1 < self.ir_len:
                    ir_len_tmp = min(self.ir_len-start_index_1, ir_tmp.shape[0])
                    ir[start_index_1: start_index_1+ir_len_tmp] = \
                        ir[start_index_1: start_index_1+ir_len_tmp] + ir_tmp[:ir_len_tmp]
                else:
                    # the remaining delay is larger than ir_len, give up
                    logging.warning(f'too larger delay   {delay_sample_num}')
            else:
                logging.warning('give up small value')

        # High-pass filtering
        # when interpolating the spectrum of absorption, DC value is assigned to the value of 125Hz
        ir = self.HP_filter(ir)

        return ir

    def make_file_logger(self, log_path):
        logger = logging.getLogger()
        file_handler = logging.FileHandler(log_path)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        return logger, file_handler

    def cal_ir_mic(self):
        """
        calculate rir with image sources already calculated
        """
        os.makedirs('img/ir', exist_ok=True)

        ir_all = []
        for mic_i, mic in enumerate(self.receiver.mic_all):
            logger, file_handler = self.make_file_logger(f'log/cal_ir_mic_{mic_i}.log')
            ir = self._cal_ir_1mic(mic, logger)
            ir_all.append(ir.reshape([-1, 1]))
            logger.removeHandler(file_handler)
        ir_all = np.concatenate(ir_all, axis=1)
        return ir_all

    def cal_ir_reciver(self):
        logger = self.make_file_logger('log/cal_ir_receiver.log')
        ir = self._cal_ir_1mic(self.receiver, logger)
        return ir
    
    def save_img_info(self, data_path=None):
        if data_path is None:
            data_path = 'tmp/img_info.npz'
        np.savez(data_path, 
                 n_img=self.n_img,
                 img_pos_all=self.img_pos_all, 
                 reflect_attenuate_all=self.reflect_attenuate_all)

    def load_img_info(self, data_path=None):
        if data_path is None:
            data_path = 'tmp/img_info.npz'
        info = np.load(data_path)
        self.n_img = info['n_img']
        self.img_pos_all = info['img_pos_all']
        self.reflect_attenuate_all=info['reflect_attenuate_all']


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config['Basic'] = {'Fs': 44100,
                       'reflect_order': -1,
                       'HP_cutoff': 100}
    config['Room'] = {'size': '4, 4, 4',
                      'RT60': ', '.join([f'{item}' for item in np.ones(6) * 0.1]),
                      'A': ''}
    config['Source'] = {'pos': '3, 2, 2',
                        'view': '0, 0, 0',
                        'directivity':'omnidirectional'}
    config['Receiver'] = {'pos': '2, 2, 2',
                          'view': '0, 0, 0',
                          'n_mic': '2'}
    head_r = 0.145/2
    config['Mic_0'] = {'pos': f'0, {head_r}, 0',
                       'view': '-90, 0, 0',
                       'direct_type': 'binaural_L'}
    config['Mic_1'] = {'pos': f'0, {-head_r}, 0',
                       'view': '90, 0, 0',
                       'direct_type': 'binaural_R'}

    # logging.basicConfig(level=logging.INFO)
    roomsim = RoomSim(config)

    # fig, ax = roomsim.show()
    # plt.show()

    roomsim.get_img()
    # roomsim.save_img_info()

    # roomsim.load_img_info()
    rir = roomsim.cal_ir_mic()

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(rir[:, 0])
    ax[1].plot(rir[:, 1])
    plt.show()

    np.save('rir.npy', rir)