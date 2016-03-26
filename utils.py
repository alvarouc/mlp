import matplotlib.pyplot as plt
import numpy as np


def diagnostic_plot(clf, file_name):
    plt.figure()
    params = {'alpha': 0.7, 'linestyle': 'solid', 'marker': '.'}
    loss = clf.history.history['loss']
    plt.plot(np.log10(loss), label='Train loss', **params)
    if 'val_loss' in clf.history.history.keys():
        val_loss = clf.history.history['val_loss']
        plt.plot(np.log10(val_loss), label='Validation loss', **params)
    if hasattr(clf, 'test_loss'):
        test_loss = clf.test_loss.test_losses
        plt.plot(np.log10(test_loss), label='Test loss', **params)
    plt.legend()
    plt.ylabel('$\mathrm{Log}_{10}(\mathrm{loss})$')
    plt.xlabel('Epoch')
    plt.savefig(file_name+'.pdf')
