import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import tango


def smooth(x, n=10):
    return np.convolve(x, np.ones((n,))/n, mode='valid')


def plot(df, color, label):
    y = smooth(df['epoch/return'])
    # x = df['total/episodes'][:len(y)]
    x = df['total/frames'][:len(y)]
    std = smooth(df['epoch/return_std'] / np.sqrt(df['epoch/episodes']))

    plt.plot(x, y, color=color, label=label)
    plt.fill_between(x, y-std, y+std, color=color, alpha=0.5, linewidth=0)


if __name__ == '__main__':
    api = wandb.Api()
    runs = api.runs('bradyz/rl', {'config.algorithm': 'ppo'})
    colors = [tango.COLORS[k] for k in ['skyblue1', 'scarletred2', 'chameleon3']]
    # colors = [tango.COLORS[k] for k in ['plum3', 'indigo', 'butter3']]

    wanted = ['stellar-voice-211', 'misunderstood-puddle-212', 'major-bush-210']
    key = 'iterations'

    wanted = ['valiant-violet-195', 'grateful-blaze-202', 'swept-spaceship-208']
    key = 'gamma'

    # wanted = ['fallen-smoke-209', 'glowing-rain-201', 'dark-moon-207']
    # key = 'gamma'

    plt.figure(figsize=(4, 3))

    for j, name in enumerate(wanted):
        for i, run in enumerate(runs):
            if run.name != name:
                continue

            df = pd.DataFrame(run.history(samples=int(1e9)))
            df = df[df['epoch/return'].notnull()]

            plot(df, colors[j], r'$\gamma = %.2f$' % run.config['gamma'])
            # plot(df, colors[j], r'steps = %d' % run.config[key])

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % (x / 1e6)))
    # g = lambda x, pos: '%d' % (x / 1e3)

    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))
    plt.title('Hacienda')
    plt.xlabel('Millions of Frames')
    # plt.xlabel('Thousands of Episodes')
    # plt.xlim(0, 10000)
    plt.xlim(0, 6e6)
    plt.ylabel('Fraction of Track Traveled')
    plt.ylim(0, 1)
    plt.grid()
    plt.tight_layout()
    plt.legend(loc='lower right')
    # plt.legend(loc='upper left')
    plt.savefig('discount_hacienda.pdf', dpi=300)
