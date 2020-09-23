import numpy as np
from .Reverb import RT2Absorb, Absorb2RT
from .plot_cube import plot_cube


class ShoeBox(object):
    def __init__(self, config):
        self.F_abs = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
        self._load_config(config)
        self.size_double = 2*self.size

    def _load_config(self, config):
        self.size = np.asarray([np.float32(item)
                                for item in config['size'].split(',')])
        if len(config['RT60']) < 1:
            self.RT60 = None
        else:
            self.RT60 = np.asarray([np.float32(item)
                                    for item in config['RT60'].split(',')])

        if len(config['A']) < 1:
            self.A = RT2Absorb(self.RT60, self.size)
        else:
            self.A = np.asarray(
                [np.float32(item)
                 for item in config['A'].split(',')]).reshape([6, 7])

        if self.RT60 is None:
            self.RT60 = Absorb2RT(self.A, room_size=self.size)
        self.B = np.sqrt(1 - self.A)  # reflection coefficients

    def visualized(self, ax=None, extra_point=None, show_absorption=False):
        if not show_absorption:
            fig, ax = plot_cube(self.size, ax, extra_point)
        else:
            fig, ax = plot_cube(self.size, ax, extra_point,
                                bright_color_all=np.mean(self.A, axis=1))
        return fig, ax

    def visualize_xy(self, ax, fig_path=None):
        wall_plot_settings = {'color': 'black',
                              'linewidth': 1}
        ax.plot([0, 0], [self.size[0], 0], **wall_plot_settings)
        ax.plot([self.size[0], 0], [self.size[0], self.size[1]],
                **wall_plot_settings)
        ax.plot([self.size[0], self.size[1]], [0, self.size[1]],
                **wall_plot_settings)
        ax.plot([0, self.size[1]], [0, 0], **wall_plot_settings)


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config['Room'] = {'size': '4 5 6',
                      'RT60': ' '.join([f'{item}' for item in np.ones(6)*0.5]),
                      'A': ''}
    room = ShoeBox(config['Room'])
    room.show(fig_path='img/ShoeBox_test.png', show_absorption=True)
