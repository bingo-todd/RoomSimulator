import numpy as np


B_power_table = {}


def cal_power(x, n):

    y = np.power(x, n)
    return y
    base_str = f'{x:0>.4f}'
    exp_str = f'{n:d}'
    if base_str in B_power_table.keys():
        if exp_str in B_power_table[base_str]:
            y = B_power_table[base_str][exp_str]
        else:
            y = np.power(x, n)
            # y_all = B_power_table[base_str]
            # exp_all = list(map(int, y_all.keys()))
            # # break n into parts
            # n_remain = n
            # y_tmp = 1  # temporary result
            # while len(exp_all) > 0 and n_remain > exp_all[0]:
            #     # remove too large exp
            #     exp_all = list(
            #         filter(lambda x: x <= n_remain, exp_all))
            #     n_sub = exp_all[-1]
            #     y_tmp = y_tmp * y_all[f'{n_sub}']
            #     n_remain = n_remain - n_sub
            # if n_remain < 0:
            #     print('negative exponent')
            #     return None
            # y = y_tmp * (x**n_remain)
            # y = np.power(x, n)
    else:
        y = np.power(x, n)
        B_power_table[base_str] = {}
        B_power_table[base_str][exp_str] = y
    return y


def main():
    x = 0.65
    n_all = np.random.randint(1, 2000, 10000)

    import time
    t_start = time.time()
    [cal_power(x, n) for n in n_all]
    print(n_all[:20])
    t_elapsed = time.time() - t_start
    print(f't_elapsed: {t_elapsed}')


if __name__ == '__main__':
    main()
