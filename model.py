from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adadelta
from keras.regularizers import l1
from keras.callbacks import EarlyStopping

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score

import numpy as np


class Model:
    '''
    Model class that wraps Keras model

    Input:
    in_dim: number of variables of the data
    out_dim: number of classes to predict
    n_hidden: number of hidden variables at each layers
    n_deep: number of layers
    l1_norm: penalization coefficient for L1 norm on hidden variables
    drop: dropout percentage at each layer
    verbose: verbosity level (up to 3 levels)

    Methods:
    reset_weigths: re-initiallizes the weights
    save/load: saves/loads weights to a file
    fit: trains on data provided with early stopping
    train_batch: trains on batch provided splitting
                 the data into train and validation
    fit_batches: trains on a sequence of batches with early stopping
    predict: returns prediction on data provided
    auc: returns area under the roc curve on data and true
         labels provided
    '''
    def __init__(self, in_dim, out_dim=1,
                 n_hidden=1000, n_deep=4,
                 l1_norm=0.01,
                 drop=0.1,
                 verbose=2):
        self.n_train = 1
        self.verbose = verbose
        self.model = Sequential()
        # Input layer
        self.model.add(Dense(
            input_dim=in_dim,
            output_dim=n_hidden,
            init='glorot_uniform',
            activation='tanh',
            W_regularizer=l1(l1_norm)))
        self.model.add(Dropout(drop))
        # do X layers
        for layer in range(n_deep-1):
            self.model.add(Dense(
                output_dim=np.round(n_hidden/2**(layer+1)),
                init='glorot_uniform',
                activation='tanh',
                W_regularizer=l1(l1_norm)))
            self.model.add(Dropout(drop))
        # Output layer
        self.model.add(Dense(out_dim,
                             init='glorot_uniform',
                             activation='tanh'))

        temp = [layer['output_dim']
                for layer in self.model.get_config()['layers']
                if layer['name']=='Dense']
        
        print('Model:{}, {}'.format(in_dim, temp))
        
        # Optimization algorithms
        opt = Adadelta()
        if out_dim == 1:
            self.model.compile(loss='binary_crossentropy',
                               optimizer=opt,
                               class_mode='binary')
        self.W0 = self.model.get_weights()

    def reset_weigths(self):
        self.model.set_weights(self.W0)
        self.n_train = 1

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def fit(self, x_all, y_all, verbose=0, patience=200):

        sss = StratifiedShuffleSplit(y_all, 1,
                                     test_size=0.1,
                                     random_state=0)
        for train_index, val_index in sss:
            x_train, x_val = x_all[train_index, :], x_all[val_index, :]
            y_train, y_val = y_all[train_index], y_all[val_index]

        stop = EarlyStopping(monitor='val_loss',
                             patience=patience,
                             verbose=1)
        self.model.fit(x_train, y_train,
                       nb_epoch=5000,
                       batch_size=16,
                       verbose=self.verbose,
                       callbacks=[stop],
                       show_accuracy=True,
                       validation_data=(x_val, y_val))

    def train(self, batch, label, val_data=None):
        if val_data:
            train_loss, train_accuracy\
                = self.model.train_on_batch(batch, label,
                                            accuracy=True)
            val_loss, val_accuracy\
                = self.model.evaluate(val_data[0], val_data[1],
                                      show_accuracy=True,
                                      verbose=0)
        else:
            sss = StratifiedShuffleSplit(label, 1,
                                         test_size=0.1,
                                         random_state=0)
            for train_index, val_index in sss:
                x_train, x_val = batch[train_index, :], batch[val_index, :]
                y_train, y_val = label[train_index], label[val_index]

            train_loss, train_accuracy\
                = self.model.train_on_batch(x_train, y_train,
                                            accuracy=True)
            val_loss, val_accuracy\
                = self.model.evaluate(x_val, y_val,
                                      show_accuracy=True,
                                      accuracy=True)

        if self.verbose and self.n_train == 1:
                print('Starting batch training: (loss, acuracy)')

        if self.verbose and (self.n_train % 20 == 0):
            print('Batch: %d: Train (%.3f,%.1f%%), Val (%.3f,%.1f%%)' %
                  (self.n_train,
                   train_loss, train_accuracy*100,
                   val_loss, val_accuracy*100))

        self.n_train += 1

        return(train_loss, train_accuracy, val_loss, val_accuracy)

    def fit_batches(self, batches, batches_label,
                    val_data=None, scaler=None,
                    patience=200):
        # Fit the model from a source of batches
        # batches: python iterator that generates the batches
        # batches_label: labels for all batches
        best_loss = np.infty
        current_patience = 0
        for batch in batches:
            if scaler:
                batch = scaler.transform(batch)

            train_loss, train_accuracy,\
                val_loss, val_accuracy\
                = self.train(batch, batches_label,
                             val_data=val_data)
            if val_loss <= best_loss:
                current_patience = 0
                best_loss = val_loss
                if self.verbose:
                    print('Best val_loss : %.3f' % best_loss)
            else:
                current_patience += 1

            if current_patience > patience:
                break

    def predict(self, x_all):
        prediction = self.model.predict_classes(x_all,
                                                verbose=self.verbose)
        return(prediction)

    def auc(self, x_all, y_all):
        prediction = self.model.predict(x_all,
                                        verbose=self.verbose)
        return roc_auc_score(y_all, prediction)

    def f1(self, x_all, y_all):
        prediction = self.model.predict(x_all,
                                        verbose=self.verbose)
        return f1_score(y_all, prediction)
