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
from utils import cal_dist, cartesian2pole, pole2cartesian, filter, My_Logger
from BasicTools import ProcessBar

class RoomSim(object):
    def __init__(self, room_config, source_config=None, receiver_config=None):
        """split configuration into 3 parts, which is more flexible
            room_config: room size, absorption coefficients and other basic configuration like Fs
            source_config: configuration related to source
            receiver_config: configuration related to receiver
        """
        self._load_room_config(room_config)
        self.load_source_config(source_config)
        self.load_receiver_config(receiver_config)

        self.T_Fs = 1. / self.Fs
        self.F_nyquist = self.Fs/2.0  # Half of sampling frequency

        # constants
        self.F_abs = np.array([125, 250, 500, 1000, 2000, 4000])  # frequency of each sound absorption coefficients
        self.n_F_abs = len(self.F_abs)
        self.F_abs_norm = self.F_abs/self.F_nyquist
        self.F_abs_extend = np.concatenate(([0], self.F_abs, [self.F_nyquist]))
        self.F_abs_extend_norm = self.F_abs_extend/self.F_nyquist
        self.Two_pi = 2*np.pi

        self.c_air = 343.0  # sound speed in air
        self.Fs_c = self.Fs/self.c_air  # Samples per metre
        m_air = 6.875e-4*(self.F_abs/1000)**1.7
        self.air_attenuate_per_dist = np.exp(-0.5*m_air).T

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
        max_direct_delay = cal_dist(self.room.size) / self.c_air
        self.ir_len = np.int(np.floor((np.max(RT60)+max_direct_delay)*self.Fs))
        
        self.n_cube_xyz = np.ceil(self.ir_len/self.Fs_c/(self.room.size*2))
        self.n_img_in_cube = 8  # number of image source in one cube
        # codes the eight permutations of x+/-xp, y+/-yp, z+/-zp
        # where [-1 -1 -1] identifies the parent source.
        self.rel_img_pos_in_cube = np.array(
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
        # Includes/excludes bx, by, bz depending on 0/1 state.
        self.rel_refl_num_in_cube = np.array(
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

        # init
        self.amp_gain_reflect_all = np.zeros((0, self.F_abs.shape[0]))
        self.img_pos_all = np.zeros((0, 3))
        self.reflect_gain_all = np.zeros((0, self.F_abs.shape[0]))
        self.n_img = 0

        self.B_power_table_path = 'B_power_table.npy'
        if os.path.exists(self.B_power_table_path):
            self.B_power_table = np.load(self.B_power_table_path, allow_pickle=True).item()
        else:
            self.B_power_table = {}

    def _load_room_config(self, config):
        # basic configuration
        config = config['Room']
        self.Fs = np.int32(config['Fs'])
        self.HP_cutoff = np.float(config['HP_cutoff'])
        self.reflect_order = np.int32(config['reflect_order'])
        # threshold of reflection 
        if 'amp_theta' in config.keys():
            self.amp_theta = config['amp_theta']
        else:
            self.amp_theta = 1e-6
        
        # configure of room
        self.room = ShoeBox(config)

    def load_source_config(self, config):
        if config is None:
            self.source = None
        else:
            config = config['Source']
            config['Fs'] = f'{self.Fs}'
            self.source = Source(config)
        
    def load_receiver_config(self, config):
        if config is None:
            self.receiver = None
        else:
            config['Receiver']['Fs'] = f'{self.Fs}'
            self.receiver = Receiver(config)

    def show(self):
        """"""
        fig, ax = self.room.show()
        ax.scatter(*self.source.pos, 'ro', label='source')
        self.receiver.show(ax, arrow_len=0.5)
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

    def local_power(self, x, n):
        key = f'{x}_{n}'
        if key in self.B_power_table.keys():
            y = self.B_power_table[key]
        else:
            y = np.power(x, n)
            self.B_power_table[key] = y
        return y

    def cal_wall_attenuate(self, n_refl):
        attenuate = np.ones(self.n_F_abs)
        for wall_i in range(6):
            for freq_i in range(self.n_F_abs):
                attenuate[freq_i] = attenuate[freq_i] * self.local_power(self.room.B[wall_i,freq_i], n_refl[wall_i])
        return attenuate

    def cal_all_img(self, is_plot=False, is_verbose=False, log_path='log/get_img.log'):

        # TODO input argments validate
        # use dictionary to save the exponent of B
        # init

        # Maximum number of image sources
        n_img_max = np.int(np.prod(2*self.n_cube_xyz+1)*8)
        img_pos_all = np.zeros((n_img_max, 3))  # image source co-ordinates
        reflect_gain_all = np.zeros((n_img_max, self.n_F_abs))
        n_img = -1  # number of significant images of each parent source

        img_pos_in_cube0 = self.rel_img_pos_in_cube*self.source.pos[np.newaxis, :]
        logger = My_Logger(log_path)
        for cube_i_x in np.arange(-self.n_cube_xyz[0], self.n_cube_xyz[0]+1):
            cube_pos_x = cube_i_x * self.room.size_double[0]
            for cube_i_y in np.arange(-self.n_cube_xyz[1], self.n_cube_xyz[1]+1):
                cube_pos_y = cube_i_y * self.room.size_double[1]
                for cube_i_z in np.arange(-self.n_cube_xyz[2], self.n_cube_xyz[2]+1):
                    cube_pos_z = cube_i_z * self.room.size_double[2]
                    cube_pos = np.asarray([cube_pos_x, cube_pos_y, cube_pos_z])
                    n_reflect_xyz1 = np.abs(np.asarray([cube_i_y, cube_i_x, cube_i_z]))
                    
                    for i in np.arange(self.n_img_in_cube):
                        n_img = n_img+1
                        img_pos_all[n_img] = cube_pos + img_pos_in_cube0[i]

                        n_refl_xyz0 = np.abs(np.asarray([cube_i_y, cube_i_x, cube_i_z]) - self.rel_refl_num_in_cube[i])
                        n_refl = np.concatenate((n_refl_xyz0, n_reflect_xyz1))
                        reflect_gain_all[n_img] = self.cal_wall_attenuate(np.squeeze(n_refl))
                        if np.sum(reflect_gain_all[n_img]) < self.amp_theta:
                            n_img = n_img -1
                            continue

                        logger.info(f'{n_img} {img_pos_all[n_img]}  {n_refl}')

        n_img = n_img + 1
        self.img_pos_all = img_pos_all[:n_img]
        self.reflect_gain_all = reflect_gain_all[:n_img]
        self.n_img = n_img

        # save for another run
        np.save(self.B_power_table_path, self.B_power_table)
        logger.close()

        if is_plot:
            fig, ax = self.room.show(extra_point=self.img_pos_all)
            plt.title(f'n_img: {self.n_img}')
            return fig, ax

    def amp_spec_to_ir(self, refl_amp):
        # interpolated grid points
        refl_amp_extend = np.concatenate((refl_amp[:1], refl_amp, refl_amp[-1:]))
        amp_spec_half = interp1d(self.F_abs_extend_norm, refl_amp_extend)(self.freq_norm_all)
        amp_spec = np.concatenate((amp_spec_half, np.conj(np.flip(amp_spec_half[1:-1]))))
        ir = np.real(np.fft.ifft(amp_spec))
        ir = self.window * np.concatenate((ir[self.n_fft_half_valid+1:self.n_fft],
                                           ir[:self.n_fft_half_valid+1]))
        return ir

    def _cal_ir_1mic(self, mic, log_path, is_verbose=False, img_dir=None):
        logger = My_Logger(log_path)

        ir = np.zeros(self.ir_len)
        pb = ProcessBar(self.n_img)
        if is_verbose:
            refl_count = 0
            fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=[15, 6])
            ax[0].scatter(self.source.pos[0], self.source.pos[1], c='r', marker='x', label='source')
            ax[0].scatter(self.receiver.pos[0], self.receiver.pos[1], c='r', marker='o', label='mic')
            valid_img_point_label = 'valid_img'
            invalid_img_point_label = 'invalid_img'
        for img_i in np.arange(self.n_img):
            pb.update()

            if mic.direct_type == 'binaural_L' or mic.direct_type == 'binaural_R':
                pos_img_to_mic = np.matmul(self.receiver.tm, (self.img_pos_all[img_i] - self.receiver.pos))
            else:
                pos_img_to_mic = np.matmul(mic.tm_room, (self.img_pos_all[img_i] - mic.pos_room))
            *angle_img_to_mic, dist = cartesian2pole(pos_img_to_mic)
            
            logger.info(f'{img_i},{dist},{angle_img_to_mic[0]}')
            
            # 
            pos_mic_to_img = np.matmul(self.source.tm.T, (mic.pos_room - self.img_pos_all[img_i]))
            *angle_mic_to_img, _ = cartesian2pole(pos_mic_to_img)
            
            # amplitude gain of wall
            reflect_amp = self.reflect_gain_all[img_i]
            # amplitude gain of distance
            reflect_amp = reflect_amp/dist
            # amplitude gain of air
            reflect_amp = reflect_amp*(self.air_attenuate_per_dist**dist)

            # calculate ir based on amp
            ir_tmp = self.amp_spec_to_ir(reflect_amp)

            # directivity of sound source, directivity after imaged
            ir_source = self.source.get_ir(angle_mic_to_img)
            ir_tmp = filter(ir_source, 1, ir_tmp)

            refl_valid_flag = False # whether to log this sound image in verbose
            max_amp = np.max(np.abs(ir_tmp[:self.n_fft_half_valid]))
            if  max_amp > self.amp_theta:
                    # mic directivity filter
                ir_mic = mic.get_ir(angle_img_to_mic)
                if ir_mic is not None:
                    ir_tmp = filter(ir_mic, 1, np.concatenate((ir_tmp, self.zero_padd_array)))
                    
                    # parse delay into integer and fraction.
                    delay_sample_num = dist * self.Fs_c
                    delay_sample_num_int = np.int(np.round(delay_sample_num))
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
                        refl_valid_flag = True
                    else:
                        logger.warning(f'too larger delay delay(sample):{delay_sample_num:.2f}')
            else:
                logger.warning(f'too small amplitude  max_amp:{max_amp:.2E}')

            if is_verbose:
                if self.img_pos_all[img_i, 2] == self.source.pos[2]:
                    if np.max(np.abs(self.img_pos_all[img_i] - self.source.pos)) < 1e-10:
                        continue

                    if refl_valid_flag:
                        ax[0].scatter(self.img_pos_all[img_i, 0], self.img_pos_all[img_i, 1], c='b', marker='x', label=valid_img_point_label)
                        valid_img_point_label = None
                    else:
                        ax[0].scatter(self.img_pos_all[img_i, 0], self.img_pos_all[img_i, 1], c='b', marker='x', alpha=0.3, label=invalid_img_point_label)
                        invalid_img_point_label = None
                    ax[0].legend(loc='upper right')
                    ax[0].set_title('sound image')
                    ax[1].cla()
                    ax[1].plot(ir_tmp)
                    ax[1].set_xlim([0, 3000])
                    ax[1].set_title('ir_tmp')
                    ax[2].cla()
                    ax[2].plot(ir)
                    ax[2].set_xlim([0, 3000])
                    ax[2].set_title('rir_all')
                    fig.savefig(f'{img_dir}/{refl_count}.png')                    
                    refl_count = refl_count + 1
        
        # High-pass filtering
        # when interpolating the spectrum of absorption, DC value is assigned to the value of 125Hz
        # ir = self.HP_filter(ir)
        logger.close()
        return ir

    def cal_ir_mic(self, is_verbose=False, img_dir=None):
        """
        calculate rir with image sources already calculated
        """
        os.makedirs('img/ir', exist_ok=True)

        ir_all = []
        for mic_i, mic in enumerate(self.receiver.mic_all):
            log_path = f'log/cal_ir_mic_{mic_i}.log'
            if is_verbose:
                os.makedirs(f'{img_dir}/{mic_i}', exist_ok=True)
            ir = self._cal_ir_1mic(mic, log_path, is_verbose, f'{img_dir}/{mic_i}')
            ir_all.append(ir.reshape([-1, 1]))
        ir_all = np.concatenate(ir_all, axis=1)
        return ir_all

    def cal_ir_reciver(self):
        log_path = 'log/cal_ir_receiver.log'
        ir = self._cal_ir_1mic(self.receiver, logger)
        return ir
    
    def save_img_info(self, data_path=None):
        if data_path is None:
            data_path = 'tmp/img_info.npz'
        np.savez(data_path, 
                 n_img=self.n_img,
                 img_pos_all=self.img_pos_all, 
                 reflect_gain_all=self.reflect_gain_all)

    def load_img_info(self, data_path=None):
        if data_path is None:
            data_path = 'tmp/img_info.npz'
        info = np.load(data_path)
        self.n_img = info['n_img']
        self.img_pos_all = info['img_pos_all']
        self.reflect_gain_all=info['reflect_gain_all']


if __name__ == '__main__':

    room_config = configparser.ConfigParser()
    room_config['Room'] = {
        'size': '4, 4, 4',
        'RT60': ', '.join([f'{item}' for item in np.ones(6) * 0.2]),
        'A': '',
        'Fs': 44100,
        'reflect_order': -1,
        'HP_cutoff': 100}

    receiver_config = configparser.ConfigParser()
    receiver_config['Receiver'] = {
        'pos': '2, 2, 2',
        'view': '0, 0, 0',
        'n_mic': '2'}
    head_r = 0.145/2
    receiver_config['Mic_0'] = {
        'pos': f'0, {head_r}, 0',
        'view': '0, 0, 90',
        'direct_type': 'binaural_L'}
    receiver_config['Mic_1'] = {
        'pos': f'0, {-head_r}, 0',
        'view': '0, 0, -90',
        'direct_type': 'binaural_R'}

    roomsim = RoomSim(room_config=room_config, source_config=None, receiver_config=receiver_config)

    azi = 0
    azi_rad = azi/180*np.pi
    source_config = configparser.ConfigParser()
    source_config['Source'] = {
        'pos': f'{2+1*np.cos(azi_rad)}, {2+np.sin(azi_rad)}, 2',
        'view': '0, 0, 0',
        'directivity':'omnidirectional'}
    roomsim.load_source_config(source_config)
    fig, ax = roomsim.show()
    ax.view_init(elev=60, azim=30)
    fig.savefig(f'img/room.png', dpi=200)
    plt.close(fig)
         
    fig, ax = roomsim.cal_all_img(is_plot=True)
    fig.savefig('img/image_sources.png')

    rir = roomsim.cal_ir_mic(is_verbose=False, img_dir='img/verbose')
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(rir[:, 0])
    ax[1].plot(rir[:, 1])
    ax[2].plot(rir[:, 0] - rir[:, 1])
    fig.savefig('img/rir.png')

    np.save('rir.npy', rir)