import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib
import os

from LocTools.easy_parallel import easy_parallel
from .DelayFilter import DelayFilter
from .Room import ShoeBox
from .Source import Source
from .Receiver import Receiver
from .utils import cal_dist, cartesian2pole
from .utils import norm_filter, nonedelay_filter


class RoomSimulator(object):
    def __init__(self, room_config, source_config=None, receiver_config=None):
        """split configuration into 3 parts, which is more flexible
            room_config: room size, absorption coefficients and other basic
            configuration like Fs source_config: configuration related to
            source receiver_config: configuration related to receiver
        """
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

        # init
        self.amp_gain_reflect_all = np.zeros((0, self.F_abs.shape[0]))
        self.img_pos_all = np.zeros((0, 3))
        self.refl_gain_all = np.zeros((0, self.F_abs.shape[0]))
        self.n_img = 0

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

    def show(self, ax=None, is_zoom=False):
        """"""

        fig, ax = self.room.show(ax)

        ax.scatter(*self.source.pos, 'ro', label='source')
        self.receiver.show(ax, arrow_len=0.5)
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
            return None
        w = 2*np.pi*self.HP_cutoff
        r1, r2 = np.exp(-w*self.T_Fs), np.exp(-w*self.T_Fs)
        b1, b2 = -(1+r2), r2  # Numerator coefficients (fix zeros)
        a1, a2 = 2*r1*np.cos(w*self.T_Fs), -r1**2  # Denominator coefficients
        HP_gain = (1-b1+b2)/(1+a1-a2)  # Normalisation gain
        b = np.asarray([1, b1, b2])/HP_gain
        a = np.asarray([1, -a1, -a2])
        self._HP_filter_coef = [b, a]
        return b, a

    def _plot_HP_spec(self, fig_path='HP_filter.png'):
        if self._HP_filter_coef is None:
            print('the HP filter is not set')
        freqs, spec = scipy.signal.freqz(*self._HP_filter_coef,
                                         fs=self.Fs,
                                         worN=2048)
        fig, ax = plt.subplots(1, 2)
        amp_spec = 10*np.log10(np.abs(spec)+1e-20)
        ax[0].plot(freqs, amp_spec)
        index_tmp = np.argmin(np.abs(np.max(amp_spec)-3-amp_spec))
        ax[0].text(freqs[index_tmp], amp_spec[index_tmp],
                   f'{freqs[index_tmp]} {amp_spec[index_tmp]}')
        ax[0].set_title('amp')
        ax[1].plot(freqs, np.angle(spec)/np.pi)
        ax[1].set_title('angle')
        ax[1].yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter('%g $\pi$'))  # noqa: W605
        # ax[1].yaxis.set_major_locator(
        #   matplotlib.ticker.MultipleLocator(base=1.0))
        fig.savefig(fig_path)

    def HP_filter(self, x):
        if self._HP_filter_coef is None:
            return x
        else:
            return nonedelay_filter(*self._HP_filter_coef, x)

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

    def cal_all_img(self, is_plot=False, is_verbose=False):
        """ calculate all possible sound images based on matrix manipulation
        , which is more efficient
        To speed up:
        1. calculate the position and reflection number through each wall of
            each sound image.
        2. calculate amplitude gain of each sound image in the order of
            distance
        """

        img_pos_in_cube0 = (self.rel_img_pos_in_cube
                            * self.source.pos[np.newaxis, :])

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
            fig, ax = self.room.show(extra_point=self.img_pos_all)
            plt.title(f'n_img: {self.n_img}')
            return fig, ax

    def cal_all_img_direct(self, is_plot=False, is_verbose=False):
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
            fig, ax = self.room.show(extra_point=self.img_pos_all)
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

    def _cal_ir_refls(self, img_pos_batch, refl_amp_batch, mic):
        n_img = len(img_pos_batch)
        results = []
        for img_i in np.arange(n_img):
            if (mic.direct_type == 'binaural_L'
                    or mic.direct_type == 'binaural_R'):
                pos_img_to_mic = np.matmul(self.receiver.tm,
                                           (self.img_pos_all[img_i]
                                            - self.receiver.pos))
            else:
                pos_img_to_mic = np.matmul(mic.tm_room,
                                           (self.img_pos_all[img_i]
                                            - mic.pos_room))
            *angle_img_to_mic, dist = cartesian2pole(pos_img_to_mic)

            # parse delay into integer and fraction.
            delay_sample_num = dist * self.Fs_c
            delay_sample_num_int = np.int(np.round(delay_sample_num))
            delay_sample_num_frac = (delay_sample_num - delay_sample_num_int)

            start_index_ir_tmp = max(
                [self.n_fft_half_valid-delay_sample_num_int, 0])
            # if delay_sample_num_int is larger than n_fft_half_valid
            # apply the remain delay while add ir_tmp to ir_all
            start_index_ir = max(
                [delay_sample_num_int-self.n_fft_half_valid, 0])
            if start_index_ir > self.ir_len:
                continue
            #
            pos_mic_to_img = np.matmul(self.source.tm.T,
                                       (mic.pos_room
                                        - self.img_pos_all[img_i]))
            *angle_mic_to_img, _ = cartesian2pole(pos_mic_to_img)

            # amplitude gain of wall
            refl_amp = self.refl_gain_all[img_i]
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
            if max_amp > self.amp_theta:
                    # mic directivity filter
                ir_mic = mic.get_ir(angle_img_to_mic)
                if ir_mic is not None:
                    ir_tmp = norm_filter(
                        ir_mic, 1,
                        np.concatenate((ir_tmp, self.zero_padd_array)))

                    # apply fraction delay to ir_tmp
                    delay_filter_obj = DelayFilter(
                        self.Fs, delay_sample_num_frac/self.Fs)
                    ir_tmp = delay_filter_obj.filter(ir_tmp, is_padd=True)

                    # apply integer delay
                    ir_tmp = ir_tmp[start_index_ir_tmp:]
                    ir_len_tmp = min(
                        [self.ir_len-start_index_ir, ir_tmp.shape[0]])
                    results.append([start_index_ir, ir_tmp[:ir_len_tmp]])
        return results

    def _cal_ir_1mic_parallel(self, mic, n_worker):
        n_batch = n_worker
        n_img_per_batch = int(self.n_img/n_batch)
        tasks = []
        for batch_i in range(n_batch):
            i_start = batch_i*n_img_per_batch
            i_end = i_start+n_img_per_batch
            tasks.append(
                [self.img_pos_all[i_start:i_end],
                 self.refl_gain_all[i_start:i_end],
                 mic])
        results_batch_all = easy_parallel(self._cal_ir_refls,
                                          tasks,
                                          n_worker=n_worker)
        ir = np.zeros(self.ir_len)
        for results_batch in results_batch_all:
            for start_index_ir, ir_tmp in results_batch:
                ir_len_tmp = ir_tmp.shape[0]
                ir[start_index_ir: start_index_ir+ir_len_tmp] = (
                    ir[start_index_ir: start_index_ir+ir_len_tmp] + ir_tmp)
        return ir

    def _cal_ir_1mic(self, mic, is_verbose=False, img_dir=None):
        ir = np.zeros(self.ir_len)
        if is_verbose:
            refl_count = 0
            fig, ax = plt.subplots(1, 3, tight_layout=True, figsize=[15, 6])
            ax[0].scatter(self.source.pos[0], self.source.pos[1], c='r',
                          marker='x', label='source')
            ax[0].scatter(self.receiver.pos[0], self.receiver.pos[1], c='r',
                          marker='o', label='mic')
            valid_img_point_label = 'valid_img'
            invalid_img_point_label = 'invalid_img'
        for img_i in np.arange(self.n_img):

            if (mic.direct_type == 'binaural_L'
                    or mic.direct_type == 'binaural_R'):
                pos_img_to_mic = np.matmul(self.receiver.tm,
                                           (self.img_pos_all[img_i]
                                            - self.receiver.pos))
            else:
                pos_img_to_mic = np.matmul(mic.tm_room,
                                           (self.img_pos_all[img_i]
                                            - mic.pos_room))
            *angle_img_to_mic, dist = cartesian2pole(pos_img_to_mic)

            #
            pos_mic_to_img = np.matmul(self.source.tm.T,
                                       (mic.pos_room
                                        - self.img_pos_all[img_i]))
            *angle_mic_to_img, _ = cartesian2pole(pos_mic_to_img)

            # amplitude gain of wall
            refl_amp = self.refl_gain_all[img_i]
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
            if max_amp > self.amp_theta:
                    # mic directivity filter
                ir_mic = mic.get_ir(angle_img_to_mic)
                if ir_mic is not None:
                    ir_tmp = norm_filter(
                        ir_mic, 1,
                        np.concatenate((ir_tmp, self.zero_padd_array)))

                    # parse delay into integer and fraction.
                    delay_sample_num = dist * self.Fs_c
                    delay_sample_num_int = np.int(np.round(delay_sample_num))
                    delay_sample_num_frac = (delay_sample_num
                                             - delay_sample_num_int)

                    # apply fraction delay to ir_tmp
                    delay_filter_obj = DelayFilter(
                        self.Fs, delay_sample_num_frac/self.Fs)
                    ir_tmp = delay_filter_obj.filter(ir_tmp, is_padd=True)

                    # apply integer delay
                    # first shift ir_tmp which has delay capacity of
                    # n_fft_half_valid
                    start_index_ir_tmp = max(
                        [self.n_fft_half_valid-delay_sample_num_int, 0])
                    ir_tmp = ir_tmp[start_index_ir_tmp:]

                    # if delay_sample_num_int is larger than n_fft_half_valid
                    # apply the remain delay while add ir_tmp to ir_all
                    start_index_ir = max(
                        [delay_sample_num_int-self.n_fft_half_valid, 0])
                    if start_index_ir < self.ir_len:
                        ir_len_tmp = min(
                            [self.ir_len-start_index_ir, ir_tmp.shape[0]])
                        ir[start_index_ir: start_index_ir+ir_len_tmp] = \
                            (ir[start_index_ir: start_index_ir+ir_len_tmp]
                             + ir_tmp[:ir_len_tmp])
                        refl_valid_flag = True

            if is_verbose:
                if self.img_pos_all[img_i, 2] == self.source.pos[2]:
                    if np.max(np.abs(self.img_pos_all[img_i]
                                     - self.source.pos)) < 1e-10:
                        continue

                    if refl_valid_flag:
                        ax[0].scatter(self.img_pos_all[img_i, 0],
                                      self.img_pos_all[img_i, 1],
                                      c='b',
                                      marker='x',
                                      label=valid_img_point_label)
                        valid_img_point_label = None
                    else:
                        ax[0].scatter(self.img_pos_all[img_i, 0],
                                      self.img_pos_all[img_i, 1],
                                      c='b',
                                      marker='x',
                                      alpha=0.3,
                                      label=invalid_img_point_label)
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
        # when interpolating the spectrum of absorption, DC value is assigned
        # to the value of 125Hz
        ir = self.HP_filter(ir)
        return ir

    def cal_ir_mic(self, is_verbose=False, image_dir=None, parallel_type=2,
                   n_worker=16):
        """
        calculate rir with image sources already calculated
        """
        if parallel_type == 1:  # a process per mic
            tasks = []
            for mic_i, mic in enumerate(self.receiver.mic_all):
                if is_verbose:
                    os.makedirs(f'{image_dir}/{mic_i}', exist_ok=True)
                tasks.append([mic, is_verbose, f'{image_dir}/{mic_i}'])
            ir_all = easy_parallel(self._cal_ir_1mic, tasks, n_worker=n_worker)
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
            for mic in self.receiver.mic_all:
                if is_verbose:
                    os.makedirs(f'{image_dir}/{mic_i}', exist_ok=True)
                ir_all.append(
                    self._cal_ir_1mic(mic, is_verbose, f'{image_dir}/{mic_i}'))

        ir_all = np.concatenate(
            [ir.reshape([-1, 1]) for ir in ir_all],
            axis=1)
        return ir_all

    def cal_ir_reciver(self):
        ir = self._cal_ir_1mic(self.receiver)
        return ir

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
