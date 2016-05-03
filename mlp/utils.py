import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess


params = {'alpha': 0.3, 'linestyle': 'solid', 'marker': '.',
          'zorder': 1}
params_f = {'alpha': 1, 'linestyle': 'solid', 'marker': ',',
            'color': 'black', 'zorder': 2}


def filter(ts):
    temp = lowess(ts, range(len(ts)), is_sorted=True, frac=0.05, it=0)
    return temp[:, 1]


def plot_line(ts, label):
    plt.plot(np.log10(ts), label=label, **params)
    plt.plot(np.log10(filter(ts)), **params_f)
    # plt.plot(ts, label=label, **params)
    # plt.plot(filter(ts), **params_f)


def diagnostic_plot(clf, file_name=None):
    plt.figure(figsize=(9, 4))
    loss = clf.history['loss']
    plot_line(loss, 'Train loss')
    if 'val_loss' in clf.history.keys():
        val_loss = clf.history['val_loss']
        plot_line(val_loss, 'Validation loss')
    if hasattr(clf, 'test_loss'):
        test_loss = clf.test_loss.test_losses
        plot_line(test_loss, 'Test loss')
    plt.legend()
    plt.ylabel('$\mathrm{Log}_{10}(\mathrm{loss})$')
    plt.xlabel('Epoch')
    if file_name is not None:
        plt.savefig(file_name+'.pdf')
