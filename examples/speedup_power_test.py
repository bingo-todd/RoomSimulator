import numpy as np


B = np.random.rand(6, 6)


def cal_refl_amp_spec_matrix(refl_num_all):
    result = np.prod(B**(np.reshape(refl_num_all, [6, 1])), axis=0)


def cal_refl_amp_spec(refl_num_all):
    result = np.zeros(6)
    for wall_i in range(6):
        for band_i in range(6):
            result[band_i] = result[band_i] * B[wall_i, band_i]**refl_num_all[wall_i]
    return result


def main():
    import time

    t_start = time.time()
    for i in range(5000):
        refl_num_all = np.random.randint(20, 100, 6)
        cal_refl_amp_spec(refl_num_all)
    t_elapsed = time.time() - t_start
    print(f't_elapsed: {t_elapsed}')

    t_start = time.time()
    for i in range(5000):
        refl_num_all = np.random.randint(20, 100, 6)
        cal_refl_amp_spec_matrix(refl_num_all)
    t_elapsed = time.time() - t_start
    print(f't_elapsed: {t_elapsed}')



if __name__ == '__main__':
    main()
