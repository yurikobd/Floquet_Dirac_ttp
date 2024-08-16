#### pyplot style
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.collections import LineCollection
import numpy as np
from types import SimpleNamespace


# Pauli matrices
pauli = SimpleNamespace(
    s0=np.array([[1.0, 0.0], [0.0, 1.0]]),
    sx=np.array([[0.0, 1.0], [1.0, 0.0]]),
    sy=np.array([[0.0, -1j], [1j, 0.0]]),
    sz=np.array([[1.0, 0.0], [0.0, -1.0]]),
)


## Parameters and plots functions
def plotParams(pub):
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.family'] = 'STIXGeneral'
    if pub == 'paper':
        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 20, 24, 30
    elif pub == 'pres':
        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 30, 34, 40
    else:
        SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 24, 26, 30
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axis title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)


lAA = r'$~\mathrm{[\AA^{-1}]}$'
leV = r'$~\mathrm{[eV]}$'


def createline(x, y, c, normc=[-1, 1], cmap='bwr', norm=False, lw=1):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if not norm:
        norm = plt.Normalize(normc[0], normc[1])
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lw)
    lc.set_array(c)
    return lc


def createletters(axs, Axis3D=[], coord=[0.08, 0.92], ni=0, SMALL_SIZE=20):
    letters = [r'\textbf{(a)}', r'\textbf{(b)}', r'\textbf{(c)}', r'\textbf{(d)}', r'\textbf{(e)}', r'\textbf{(f)}']
    for n, ax in enumerate(axs):
        if n in Axis3D:
            ax.text2D(coord[0], coord[1], letters[max([n, ni])], transform=ax.transAxes, size=SMALL_SIZE, weight='bold')
        else:
            ax.text(coord[0], coord[1], letters[max([n, ni])], transform=ax.transAxes, size=SMALL_SIZE, weight='bold')
