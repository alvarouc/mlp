from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adadelta
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from data_generator import DataGeneratorByGroup


class BaseMLP(BaseEstimator, ClassifierMixin):
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
    def __init__(self, n_hidden=1000, n_deep=4,
                 l1_norm=0.01, drop=0.1,
                 patience=200, verbose=2):
        self.n_hidden = n_hidden
        self.n_deep = n_deep
        self.l1_norm = l1_norm
        self.drop = drop
        self.patience = patience
        self.verbose = verbose

    def fit(self, X, y, **kwargs):
        n_class = len(np.unique(y))
        if n_class == 2:
            out_dim = 1
        else:
            out_dim = n_class
        self.model = build_model(in_dim=X.shape[1], out_dim=out_dim,
                                 n_hidden=self.n_hidden, l1_norm=self.l1_norm,
                                 n_deep=self.n_deep, drop=self.drop)
        # save initial weights
        self.W0 = self.model.get_weights()

        if self.verbose:
            temp = [layer['output_dim']
                    for layer in self.model.get_config()['layers']
                    if layer['name'] == 'Dense']
            print('Model:{}'.format(temp))
        return self

    def reset_weigths(self):
        self.model.set_weights(self.W0)

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

    def predict_proba(self, X):
        return self.model.predict(X, verbose=self.verbose)

    def predict(self, X):
        prediction = self.model.predict_classes(X, verbose=self.verbose)
        return(prediction)

    def auc(self, X, y):
        prediction = self.predict(X)
        return roc_auc_score(y, prediction)

    def f1(self, X, y):
        n_class = len(np.unique(y))
        prediction = self.predict(X)
        if n_class > 2:
            return f1_score(y, prediction, average='weighted')
        else:
            return f1_score(y, prediction)


class MLP(BaseMLP):

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_class = len(np.unique(y))
        if n_class > 2:
            v_all = np.array([np.roll([1] + [0]*(n_class-1), pos)
                              for pos in y.astype('int')])
        sss = StratifiedShuffleSplit(y, 1, test_size=0.1,
                                     random_state=0)
        train_index, val_index = next(iter(sss))
        x_train, x_val = X[train_index, :], X[val_index, :]
        if n_class == 2:
            y_train, y_val = y[train_index], y[val_index]
        else:
            y_train, y_val = v_all[train_index, :], v_all[val_index, :]

        stop = EarlyStopping(monitor='val_loss',
                             patience=self.patience,
                             verbose=self.verbose)
        self.model.fit(x_train, y_train,
                       nb_epoch=5000,
                       batch_size=16,
                       verbose=self.verbose,
                       callbacks=[stop],
                       show_accuracy=True,
                       validation_data=(x_val, y_val))

        return self


class MLPg(BaseMLP):

    def __init__(self, method='ica',
                 n_components=10, n_hidden=1000, n_deep=4,
                 l1_norm=0.01, drop=0.1,
                 patience=200, verbose=2):
        self.n_hidden = n_hidden
        self.n_deep = n_deep
        self.l1_norm = l1_norm
        self.drop = drop
        self.patience = patience
        self.verbose = verbose
        self.method = method
        self.n_components = n_components

    def fit(self, X, y, scaler=None):
        # Fit the model from a source of batches
        # batches: python iterator that generates the batches
        # batches_label: labels for all batches
        super().fit(X, y)
        self.classes_, y = np.unique(y, return_inverse=True)

        batches = DataGeneratorByGroup(
            X, y, n_components=self.n_components, n_batches=1000,
            method='rejective', decomposition_method=self.method)
        batches_y = batches.batch_label
        best_loss = np.infty
        current_patience = 0
        for batch in batches:
            if scaler:
                batch = scaler.transform(batch)

            self.model.train_on_batch(batch, batches_y)
            loss_val = self.model.evaluate(X, y, verbose=0)

            if loss_val <= best_loss:
                current_patience = 0
                best_loss = loss_val
                if self.verbose:
                    print('Best val_loss : %.3f' % best_loss)
            else:
                current_patience += 1
            if current_patience > self.patience:
                break


def build_model(in_dim, out_dim=1,
                n_hidden=100, l1_norm=0.0,
                n_deep=5, drop=0.1):
    model = Sequential()
    # Input layer
    model.add(Dense(
        input_dim=in_dim,
        output_dim=n_hidden,
        init='glorot_uniform',
        activation='tanh',
        W_regularizer=l1(l1_norm)))

    # do X layers
    for layer in range(n_deep-1):
        model.add(Dropout(drop))
        model.add(Dense(
            output_dim=np.round(n_hidden/2**(layer+1)),
            init='glorot_uniform',
            activation='tanh',
            W_regularizer=l1(l1_norm)))

    # Output layer
    if out_dim == 1:
        activation = 'tanh'
    else:
        activation = 'softmax'

    model.add(Dense(out_dim,
                    init='glorot_uniform',
                    activation=activation))

    # Optimization algorithms
    opt = Adadelta()
    if out_dim == 1:
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      class_mode='binary')
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      class_mode='categorical')

    return model
