import matplotlib.pyplot as plt


def diagnostic_plot(clf, file_name):
    loss = clf.history.history['loss']
    plt.plot(loss, '.', label='Train loss')
    if 'val_loss' in clf.history.history.keys():
        val_loss = clf.history.history['val_loss']
        plt.plot(val_loss, '.', label='Validation loss')
    if hasattr(clf, 'test_loss'):
        test_loss = clf.test_loss.test_losses
        plt.plot(test_loss, '.', label='Test loss')
