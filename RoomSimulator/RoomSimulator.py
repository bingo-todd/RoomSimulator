import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
import configparser
import pickle
import os

from BasicTools.easy_parallel import easy_parallel
from BasicTools.reverb.cal_DRR import cal_DRR
from .Room import ShoeBox
from .Source import Source
from .Receiver import Receiver
from .utils import cal_dist, cartesian2pole
from .utils import norm_filter, nonedelay_filter


class RoomSimulator(object):
    def __init__(self, config_path=None,
                 room_config=None, source_config=None, receiver_config=None,
                 parent_pid=None):
        """split configuration into 3 parts, which is more flexible
            room_config: room size, absorption coefficients and other basic
            configuration like Fs source_config: configuration related to
            source receiver_config: configuration related to receiver
            parent_pid: pid used for define dump_dir
        """
        if config_path is not None:
            [room_config,
             receiver_config,
             source_config] = self.parse_config_file(config_path)

        self._load_room_config(room_config)
        self.load_source_config(source_config)
        self.load_receiver_config(receiver_config)

        self.T_Fs = 1. / self.Fs
        self.F_nyquist = self.Fs/2.0  # Half of sampling frequency

        # constants
        # frequency of each sound absorption coefficients
        self.F_abs = np.array([125, 250, 500, 1000, 2000, 4000])
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
        self._make_HP_filter()

        # calculate ir single reflection based on spectrum amplitude
        n_fft = 512  # np.int32(2*np.round(self.F_nyquist/self.F_abs[1]))
        if np.mod(n_fft, 2) == 1:
            n_fft = n_fft + 1
        self.n_fft = n_fft
        self.n_fft_half_valid = np.int32(self.n_fft/2)  # no dc component
        self.freq_norm_all = (np.arange(self.n_fft_half_valid+1)
                              / self.n_fft_half_valid)
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

        self.dump_dir = f'dump/RoomSimulator_{parent_pid}'

        self.delay_ir_dir = f'{self.dump_dir}/delay_filter'
        os.makedirs(self.delay_ir_dir, exist_ok=True)

        self.B_power_table_path = f'{self.dump_dir}/B_power_table.pkl'
        if os.path.exists(self.B_power_table_path):
            try:
                with open(self.B_power_table_path, 'rb') as B_power_table_file:
                    self.B_power_table = pickle.load(B_power_table_file)
            except Exception:
                os.system(f'rm {self.B_power_table_path}')
                self.B_power_table = {}
        else:
            self.B_power_table = {}

        self.cube_index_dir = f'{self.dump_dir}/cube_index'
        os.makedirs(self.cube_index_dir, exist_ok=True)

        # init
        self.amp_gain_reflect_all = np.zeros((0, self.F_abs.shape[0]))
        self.img_pos_all = np.zeros((0, 3))
        self.refl_gain_all = np.zeros((0, self.F_abs.shape[0]))
        self.n_img = 0

    def clean_dump(self):
        os.system(f'rm -r {self.dump_dir}')

    def parse_config_file(self, config_path):
        config_all = configparser.ConfigParser()
        config_all.read(config_path)

        room_config = configparser.ConfigParser()
        room_config['Room'] = config_all['Room']

        receiver_config = configparser.ConfigParser()
        receiver_config['Receiver'] = config_all['Receiver']
        n_mic = int(receiver_config['Receiver']['n_mic'])
        for mic_i in range(n_mic):
            receiver_config[f'Mic_{mic_i}'] = config_all[f'Mic_{mic_i}']

        source_config = configparser.ConfigParser()
        source_config['Source'] = config_all['Source']
        return room_config, receiver_config, source_config

    def _load_room_config(self, config):
        # basic configuration
        config = config['Room']
        self.Fs = np.int32(config['Fs'])
        if 'HP_cutoff' in config.keys():
            self.HP_cutoff = np.float(config['HP_cutoff'])
        else:
            self.HP_cutoff = None
        self.reflect_order = np.int32(config['reflect_order'])
        # threshold of reflection
        if 'amp_theta' in config.keys():
            self.amp_theta = config['amp_theta']
        else:
            self.amp_theta = 1e-4

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

    def visualize(self, ax=None, is_zoom=False):
        """"""

        fig, ax = self.room.visualize(ax)

        ax.scatter(*self.source.pos, 'ro', label='source')
        self.receiver.visualize(ax, arrow_len=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if is_zoom:
            min_x = min((self.receiver.pos[0], self.source.pos[0])) - 0.5
            max_x = max((self.receiver.pos[0], self.source.pos[0])) + 0.5
            min_y = min((self.receiver.pos[1], self.source.pos[1])) - 0.5
            max_y = max((self.receiver.pos[1], self.source.pos[1])) + 0.5
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

        ax.legend()
        return fig, ax

    def _make_HP_filter(self):
        # Second order high-pass IIR filter to remove DC buildup
        # (nominal -4dB cut-off at 20 Hz)
        # cutoff frequency
        if self.HP_cutoff is None:
            self._HP_filter_coef = None
        else:
            w = 2*np.pi*self.HP_cutoff
            r1, r2 = np.exp(-w*self.T_Fs), np.exp(-w*self.T_Fs)
            # Numerator coefficients (fix zeros)
            b1, b2 = -(1+r2), r2
            # Denominator coefficients
            a1, a2 = 2*r1*np.cos(w*self.T_Fs), -r1**2
            HP_gain = (1-b1+b2)/(1+a1-a2)  # Normalisation gain
            b = np.asarray([1, b1, b2])/HP_gain
            a = np.asarray([1, -a1, -a2])
            self._HP_filter_coef = [b, a]

    def _plot_HP_spec(self, fig_path='HP_filter.png'):
        if self._HP_filter_coef is None:
            print('the HP filter is not set')
        freqs, spec = scipy.signal.freqz(*self._HP_filter_coef,
                                         fs=self.Fs,
                                         worN=2048)
        fig, ax = plt.subplots(1, 3, figsize=[10, 4])
        x = np.zeros(1024)
        x[10] = 1
        ir = norm_filter(*self._HP_filter_coef, x)
        ax[0].plot(ir)
        ax[0].set_title('ir')

        amp_spec = 10*np.log10(np.abs(spec)+1e-20)
        ax[1].plot(freqs, amp_spec)
        index_tmp = np.argmin(np.abs(np.max(amp_spec)-3-amp_spec))
        ax[1].text(freqs[index_tmp], amp_spec[index_tmp],
                   f'{freqs[index_tmp]} {amp_spec[index_tmp]}')
        ax[1].set_title('amp')

        ax[2].plot(freqs, np.angle(spec)/np.pi)
        ax[2].set_title('angle')
        ax[2].yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%g $\pi$'))  # noqa: W605
        fig.savefig(fig_path)

    def HP_filter(self, x):
        if self._HP_filter_coef is None:
            return x
        else:
            return nonedelay_filter(*self._HP_filter_coef, x)

    def delay_filter(self, x, n_sample, is_padd=False,
                     padd_len=None, f_high=None, id_padd=False):
        """
        fir delay filter
        """
        order = 128

        if f_high is None:
            f_high = 0.98

        n_sample_int = np.int(np.round(n_sample))
        if n_sample_int < 0:
            x = np.pad(x[n_sample_int:], [0, n_sample_int])
        elif n_sample_int > 0:
            x = np.pad(x[:-n_sample_int], [n_sample_int, 0])
        n_sample = n_sample - n_sample_int

        if np.abs(n_sample) > 1e-10:
            ir_path = f'{self.delay_ir_dir}/{n_sample:.5f}.npy'
            try:
                ir = np.load(ir_path, allow_pickle=True)
            except Exception:
                sample_index_all = np.arange(order) - n_sample
                ir = (((f_high*order*np.sinc(f_high*sample_index_all)
                        * np.cos(np.pi/order*sample_index_all))
                       + np.cos(f_high*np.pi*sample_index_all))
                      * f_high*order/(order*order)/f_high)
                np.save(ir_path, ir)

            if is_padd:
                if padd_len is None:
                    padd_len = order
                x = np.pad(x, [0, padd_len])
            x = scipy.signal.lfilter(ir, 1, x)
        return x

    def local_power(self, x, n):
        # for efficience test
        # return np.power(x, n)

        if n < 1e-10:
            return 1

        base_str = f'{x:0>.4f}'
        exp_str = f'{n:d}'
        if (base_str in self.B_power_table.keys()
                and exp_str in self.B_power_table[base_str]):
            y = self.B_power_table[base_str][exp_str]
        else:
            # find all item with the same base
            if base_str not in self.B_power_table.keys():
                self.B_power_table[base_str] = {}
                y = np.power(x, n)
            else:
                y_all = self.B_power_table[base_str]
                exp_all = list(map(int, y_all.keys()))
                # break n into parts
                n_remain = n
                y_tmp = 1  # temporary result
                while len(exp_all) > 0 and n_remain > exp_all[0]:
                    # remove too large exp
                    exp_all = list(
                        filter(lambda x: x <= n_remain, exp_all))
                    n_sub = exp_all[-1]
                    y_tmp = y_tmp * y_all[f'{n_sub}']
                    n_remain = n_remain - n_sub
                if n_remain < 0:
                    print('negative exponent')
                    return None
                y = y_tmp * (x**n_remain)
            self.B_power_table[base_str][exp_str] = y
        return y

    def cal_wall_attenuate(self, n_refl):
        attenuate = np.ones(self.n_F_abs)
        for wall_i in range(6):
            for freq_i in range(self.n_F_abs):
                attenuate[freq_i] = (attenuate[freq_i]
                                     * self.local_power(
                                         self.room.B[wall_i, freq_i],
                                         n_refl[wall_i]))
        return attenuate

    def plot_all_img(self, ax=None):
        if ax is None:
            fig, ax = self.room.visualize(extra_point=self.img_pos_all)
        else:
            fig = None

        fig, ax = self.room.visualize(extra_point=self.img_pos_all)
        plt.title(f'n_img: {self.n_img}')
        return fig, ax

    def cal_all_img(self, is_plot=False, is_verbose=False, n_worker=1):
        """ calculate all possible sound images based on matrix manipulation
        , which is more efficient
        To speed up:
        1. calculate the position and reflection number through each wall of
            each sound image.
        2. calculate amplitude gain of each sound image in the order of
            distance
        """

        cube_index_all_path = os.path.join(
            self.cube_index_dir,
            '_'.join(map(str, self.n_cube_xyz))+'.npy')
        if not os.path.exists(cube_index_all_path):
            index_x_all, index_y_all, index_z_all = np.meshgrid(
                np.arange(-self.n_cube_xyz[0], self.n_cube_xyz[0]+1,
                          dtype=np.int16),
                np.arange(-self.n_cube_xyz[1], self.n_cube_xyz[1]+1,
                          dtype=np.int16),
                np.arange(-self.n_cube_xyz[2], self.n_cube_xyz[2]+1,
                          dtype=np.int16))
            index_x_all, index_y_all, index_z_all = (
                index_x_all.flatten().reshape(-1, 1),
                index_y_all.flatten().reshape(-1, 1),
                index_z_all.flatten().reshape(-1, 1))
            cube_index_all = np.concatenate(
                (index_x_all, index_y_all, index_z_all),
                axis=1)
            np.save(cube_index_all_path, cube_index_all)
        else:
            cube_index_all = np.load(cube_index_all_path, allow_pickle=True)

        img_pos_in_cube0 = (self.rel_img_pos_in_cube
                            * self.source.pos[np.newaxis, :])
        cube_pos_all = cube_index_all * self.room.size_double[np.newaxis, :]

        n_cube = cube_pos_all.shape[0]
        n_img = n_cube * self.n_img_in_cube
        img_pos_all = np.reshape(
            (cube_pos_all[:, np.newaxis, :]
             + img_pos_in_cube0[np.newaxis, :, :]),
            (n_img, 3))

        # number of reflections on each wall
        cube_index_all_swapxy = cube_index_all.copy()
        cube_index_all_swapxy[:, 0] = cube_index_all[:, 1]
        cube_index_all_swapxy[:, 1] = cube_index_all[:, 0]
        cube_index_all_swapxy = np.expand_dims(
            cube_index_all_swapxy, axis=1).repeat(self.n_img_in_cube, axis=1)
        n_refl_xyz1_all = np.abs(cube_index_all_swapxy)
        n_refl_xyz0_all = np.abs(cube_index_all_swapxy
                                 - self.rel_refl_num_in_cube[np.newaxis, :, :])
        n_refl_all = np.concatenate(
            (n_refl_xyz0_all, n_refl_xyz1_all),
            axis=2).reshape(n_img, 6)

        # get order according distance(block distance for efficiency)
        sort_index = np.argsort(np.sum(np.abs(img_pos_all), axis=1))
        img_pos_all = img_pos_all[sort_index]
        n_refl_all = n_refl_all[sort_index]

        if n_worker > 1:
            refl_gain_all = np.asarray(
                easy_parallel(self.cal_wall_attenuate,
                              n_refl_all[:, np.newaxis, :],
                              n_worker=n_worker,
                              dump_dir=self.dump_dir),
                dtype=np.float16)
        else:
            refl_gain_all = np.asarray(
                [self.cal_wall_attenuate(n_refl_all[img_i])
                 for img_i in range(n_img)],
                dtype=np.float16)
        valid_img_index = np.where(np.sum(refl_gain_all, axis=1)
                                   >= self.amp_theta)[0]
        self.n_img = valid_img_index.shape[0]
        self.img_pos_all = img_pos_all[valid_img_index]
        self.refl_gain_all = refl_gain_all[valid_img_index]

        if is_plot:
            fig, ax = self.room.visualize(extra_point=self.img_pos_all)
            plt.title(f'n_img: {self.n_img}')
            return fig, ax

    def cal_all_img_directly(self, is_plot=False, is_verbose=False):
        """ calculate all possible sound images using 4 layer of loop
        To speed up:
        1. calculate the position and reflection number through each wall of
            each sound image.
        2. calculate amplitude gain of each sound image in the order of
            distance
        """

        img_pos_in_cube0 = (self.rel_img_pos_in_cube
                            * self.source.pos[np.newaxis, :])
        # Maximum number of image sources
        n_img_max = np.int(np.prod(2*self.n_cube_xyz+1)*8)
        img_pos_all_tmp = np.zeros((n_img_max, 3), dtype=np.float16)
        n_refl_all_tmp = np.zeros((n_img_max, 6), dtype=np.int)
        # distance of sound image to origin
        dist_all_tmp = np.zeros((n_img_max), dtype=np.float16)

        n_img = -1
        for cube_i_x in np.arange(-self.n_cube_xyz[0], self.n_cube_xyz[0]+1):
            cube_pos_x = cube_i_x * self.room.size_double[0]
            for cube_i_y in np.arange(-self.n_cube_xyz[1],
                                      self.n_cube_xyz[1]+1):
                cube_pos_y = cube_i_y * self.room.size_double[1]
                for cube_i_z in np.arange(-self.n_cube_xyz[2],
                                          self.n_cube_xyz[2]+1):
                    cube_pos_z = cube_i_z * self.room.size_double[2]
                    cube_pos = np.asarray([cube_pos_x, cube_pos_y, cube_pos_z])
                    n_reflect_xyz1 = np.abs(
                        np.asarray(
                            [cube_i_y, cube_i_x, cube_i_z], dtype=np.int))

                    for i in np.arange(self.n_img_in_cube):
                        n_img = n_img + 1
                        img_pos_all_tmp[n_img] = cube_pos + img_pos_in_cube0[i]
                        dist_all_tmp[n_img] = np.sum(
                            np.abs(img_pos_all_tmp[n_img]))

                        n_refl_xyz0 = np.abs(
                                np.asarray([cube_i_y, cube_i_x, cube_i_z],
                                           dtype=np.int)
                                - self.rel_refl_num_in_cube[i])
                        n_refl_all_tmp[n_img] = np.concatenate(
                            (n_refl_xyz0, n_reflect_xyz1))

        # sort by distance
        sort_index = np.argsort(dist_all_tmp)
        del dist_all_tmp  # dist_all is no longer used

        img_pos_all = np.zeros((n_img_max, 3), dtype=np.float32)
        refl_gain_all = np.zeros((n_img_max, self.n_F_abs), dtype=np.float16)
        n_img = 0
        for index in sort_index:
            gain = self.cal_wall_attenuate(np.squeeze(n_refl_all_tmp[index]))
            if np.sum(gain) < self.amp_theta:
                continue
            refl_gain_all[n_img] = gain
            img_pos_all[n_img] = img_pos_all_tmp[index]
            n_img = n_img + 1

        # clear variables that are not longer used
        del n_refl_all_tmp, img_pos_all_tmp,

        self.img_pos_all = img_pos_all[:n_img]
        self.refl_gain_all = refl_gain_all[:n_img]
        self.n_img = n_img

        if is_plot:
            fig, ax = self.room.visualize(extra_point=self.img_pos_all)
            plt.title(f'n_img: {self.n_img}')
            return fig, ax

    def amp_spec_to_ir(self, refl_amp):
        # interpolated grid points
        refl_amp_extend = np.concatenate(
            (refl_amp[:1], refl_amp, refl_amp[-1:]))
        amp_spec_half = interp1d(self.F_abs_extend_norm, refl_amp_extend)(
            self.freq_norm_all)
        amp_spec = np.concatenate(
            (amp_spec_half, np.conj(np.flip(amp_spec_half[1:-1]))))
        ir = np.real(np.fft.ifft(amp_spec))
        ir = self.window * np.concatenate(
            (ir[self.n_fft_half_valid+1:self.n_fft],
             ir[:self.n_fft_half_valid+1]))
        return ir

    def _cal_ir_1refl(self, img_pos, refl_gain, mic):
        """calculate the impulse response of sound image
        Args:
            img_pos: the position of sound image
            refl_gain: the amplitude gain of reflections
            mic: microphone obj
        Returns:
            start_index_ir: the position of whole ir where this
                            ir should be added
            ir: impulse response of 1 sound image

        """
        if np.max(np.abs(img_pos-self.source.pos)) < 1e-10:
            is_direct = True
        else:
            is_direct = False

        if (mic.direct_type == 'binaural_L'
                or mic.direct_type == 'binaural_R'):
            pos_img_to_mic = np.matmul(self.receiver.tm,
                                       img_pos - self.receiver.pos)
        else:
            pos_img_to_mic = np.matmul(mic.tm_room,
                                       img_pos - mic.pos_room)
        *angle_img_to_mic, dist = cartesian2pole(pos_img_to_mic)

        # parse delay into integer and fraction.
        delay_sample_num = dist * self.Fs_c
        delay_sample_num_int = np.int(np.round(delay_sample_num))
        delay_sample_num_frac = (delay_sample_num-delay_sample_num_int)

        start_index_ir_tmp = max(
            [self.n_fft_half_valid-delay_sample_num_int, 0])
        # if delay_sample_num_int is larger than n_fft_half_valid
        # apply the remain delay while add ir_tmp to ir_all
        start_index_ir = max(
            [delay_sample_num_int-self.n_fft_half_valid, 0])

        if start_index_ir < self.ir_len:
            #
            pos_mic_to_img = np.matmul(self.source.tm,
                                       mic.pos_room - img_pos)
            *angle_mic_to_img, _ = cartesian2pole(pos_mic_to_img)

            # amplitude gain of wall
            refl_amp = refl_gain
            # amplitude gain of distance
            refl_amp = refl_amp/dist
            # amplitude gain of air
            refl_amp = refl_amp*(self.air_attenuate_per_dist**dist)

            # calculate ir based on amp
            ir_tmp = self.amp_spec_to_ir(refl_amp)

            # directivity of sound source, directivity after imaged
            ir_source = self.source.get_ir(angle_mic_to_img)
            ir_tmp = norm_filter(ir_source, 1, ir_tmp)

            max_amp = np.max(np.abs(ir_tmp[:self.n_fft_half_valid]))
            if is_direct or max_amp > self.amp_theta:
                # direct sound and reflections with amplitude exceed threshold
                # mic directivity filter
                ir_mic = mic.get_ir(angle_img_to_mic)
                if ir_mic is not None:
                    ir_tmp = norm_filter(
                        ir_mic, 1,
                        np.concatenate((ir_tmp, self.zero_padd_array)))

                    # apply fraction delay to ir_tmp
                    ir_tmp = self.delay_filter(ir_tmp, delay_sample_num_frac)

                    # apply integer delay
                    ir_tmp = ir_tmp[start_index_ir_tmp:]
                    ir_len_tmp = min(
                        [self.ir_len-start_index_ir, ir_tmp.shape[0]])
                    return start_index_ir, ir_tmp[:ir_len_tmp]
        return None

    def _cal_ir_refls(self, img_pos, refl_amp, mic):
        n_img = len(img_pos)
        results = []
        for img_i in np.arange(n_img):
            result = self._cal_ir_1refl(img_pos[img_i],
                                        refl_amp[img_i],
                                        mic)
            results.append(result)
        return results

    def _cal_ir_1mic_parallel(self, mic, n_worker):
        if self.n_img < n_worker:
            n_worker = self.n_img

        n_batch = n_worker
        n_img_per_batch = int(self.n_img/n_batch)
        img_index_perm = np.random.permutation(self.n_img)
        tasks = []
        for batch_i in range(n_batch):
            i_start = batch_i*n_img_per_batch
            i_end = i_start+n_img_per_batch
            img_index_batch = img_index_perm[i_start:i_end]
            tasks.append(
                [self.img_pos_all[img_index_batch],
                 self.refl_gain_all[img_index_batch],
                 mic])
        results_batch_all = easy_parallel(self._cal_ir_refls,
                                          tasks,
                                          n_worker=n_worker,
                                          dump_dir=self.dump_dir)
        if results_batch_all is None:
            print(f'n_img: {self.n_img}',
                  f'n_batch: {n_batch}')
            raise Exception('NULL rir')
        ir = np.zeros(self.ir_len)
        for results_batch in results_batch_all:
            for result in results_batch:
                if result is None:
                    continue
                start_index_ir, ir_tmp = result
                ir_len_tmp = ir_tmp.shape[0]
                ir[start_index_ir: start_index_ir+ir_len_tmp] = (
                    ir[start_index_ir: start_index_ir+ir_len_tmp] + ir_tmp)
        # High-pass filtering
        # when interpolating the spectrum of absorption, DC value is
        # assigned to the value of 125Hz
        ir = self.HP_filter(ir)
        return ir

    def _cal_ir_1mic(self, mic, is_verbose=False, img_dir=None):
        ir = np.zeros(self.ir_len)
        results = self._cal_ir_refls(self.img_pos_all, self.refl_gain_all, mic)
        for img_i, result in enumerate(results):
            if result is None:
                continue
            start_index, ir_tmp = result
            ir_len_tmp = ir_tmp.shape[0]
            ir[start_index: start_index+ir_len_tmp] = (
                ir[start_index: start_index+ir_len_tmp] + ir_tmp)

        # High-pass filtering
        # when interpolating the spectrum of absorption, DC value is assigned
        # to the value of 125Hz
        ir = self.HP_filter(ir)
        return ir

    def _cal_direct_ir_1mic(self, mic):
        ir = np.zeros(self.ir_len)
        start_index, ir_tmp = self._cal_ir_1refl(self.source.pos,
                                                 np.ones(6),
                                                 mic)
        ir_len_tmp = ir_tmp.shape[0]
        ir[start_index: start_index+ir_len_tmp] = (
            ir[start_index: start_index+ir_len_tmp] + ir_tmp)
        # High-pass filtering
        # when interpolating the spectrum of absorption, DC value is assigned
        # to the value of 125Hz
        ir = self.HP_filter(ir)
        return ir

    def cal_ir_mic(self, is_verbose=False, image_dir=None, parallel_type=2,
                   n_worker=8):
        """
        calculate rir with image sources already calculated
        """

        if parallel_type == 1:  # a process per mic
            tasks = []
            for mic_i, mic in enumerate(self.receiver.mic_all):
                if is_verbose:
                    os.makedirs(f'{image_dir}/{mic_i}', exist_ok=True)
                tasks.append([mic, is_verbose, f'{image_dir}/{mic_i}'])
            ir_all = easy_parallel(self._cal_ir_1mic, tasks,
                                   n_worker=n_worker, dump_dir=self.dump_dir)
        elif parallel_type == 2:
            # for each mic, divide sound images into batches
            # 1 process per batch
            ir_all = []
            for mic in self.receiver.mic_all:
                ir_all.append(
                    self._cal_ir_1mic_parallel(mic, n_worker))
        else:
            # no parallel
            ir_all = []
            for mic_i, mic in enumerate(self.receiver.mic_all):
                if is_verbose:
                    os.makedirs(f'{image_dir}/{mic_i}', exist_ok=True)
                ir_all.append(
                    self._cal_ir_1mic(mic, is_verbose, f'{image_dir}/{mic_i}'))

        ir_all = np.concatenate(
            [ir.reshape([-1, 1]) for ir in ir_all if ir is not None],
            axis=1)

        with open(self.B_power_table_path, 'wb') as B_power_table_file:
            pickle.dump(self.B_power_table, B_power_table_file)
        return ir_all

    def cal_direct_ir_mic(self):
        direct_ir_all = []
        for mic in self.receiver.mic_all:
            direct_ir = self._cal_direct_ir_1mic(mic)
            direct_ir_all.append(direct_ir)
        direct_ir_all = np.concatenate(
            [ir.reshape([-1, 1]) for ir in direct_ir_all],
            axis=1)
        return direct_ir_all

    def cal_DRR(self, n_worker=1):
        if n_worker > 1:
            ir = self._cal_ir_1mic_parallel(self.receiver, n_worker)
        else:
            ir = self._cal_ir_1mic(self.receiver)
        direct_ir = self._cal_direct_ir_1mic(self.receiver)
        drr = cal_DRR(ir, direct_ir)
        return drr

    def save_img_info(self, data_path=None):
        if data_path is None:
            data_path = 'tmp/img_info.npz'
        np.savez(data_path,
                 n_img=self.n_img,
                 img_pos_all=self.img_pos_all,
                 refl_gain_all=self.refl_gain_all)

    def load_img_info(self, data_path=None):
        if data_path is None:
            data_path = 'tmp/img_info.npz'
        info = np.load(data_path)
        self.n_img = info['n_img']
        self.img_pos_all = info['img_pos_all']
        self.refl_gain_all = info['refl_gain_all']
