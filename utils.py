import numpy as np
np.random.seed(1988)

from data_generator import DataGeneratorByGroup
from sklearn.cross_validation import StratifiedKFold as skfold
from sklearn.preprocessing import StandardScaler as sc


    
def cross_val_score(model, x_all, y_all, cv=10):
    n_class = len(np.unique(y_all))
    
    print('X: (%d,%d)' % x_all.shape)
    print('y: (%d,%d)' % (y_all.shape[0], n_class))
    
    kf = skfold(y_all, n_folds=cv)
    scores = []

    for n, (train_index, test_index) in enumerate(kf):
        model.reset_weigths()
        print('Running fold %d/%d' % (n+1, cv))
        x_train, x_test = x_all[train_index, :], x_all[test_index, :]
        y_train, y_test = y_all[train_index], y_all[test_index]
        
        model.fit(x_train, y_train)
        test_score = model.f1(x_test, y_test)
        train_score = model.f1(x_train, y_train)
        scores.append(test_score)
        print('Train %.3f, Test %.3f' % (train_score, test_score))

    return(scores)


def cross_val_score_sim(model, x_all, y_all, cv=10,
                        batch_size=100,
                        method='normal'):

    batch_per_group = int(batch_size/2)
    print('X: (%d,%d)' % x_all.shape)
    print('y: (%d,)' % y_all.shape[0])
    kf = skfold(y_all, n_folds=cv)
    scores = []

    for n, (train_index, test_index) in enumerate(kf):
        model.reset_weigths()
        print('Running fold %d/%d' % (n+1, cv))
        x_train, x_test = x_all[train_index, :], x_all[test_index, :]
        y_train, y_test = y_all[train_index], y_all[test_index]
        batches = DataGeneratorByGroup(x_train, y_train,
                                       n_components=10,
                                       method=method,
                                       n_samples=batch_per_group,
                                       n_batches=1000)
        batch_label = batches.batch_label
        # np.array([0]*batch_per_group + [1]*batch_per_group)

        scaler = sc()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        model.fit_batches(batches, batch_label,
                          val_data=(x_train, y_train),
                          scaler=scaler)
        model.fit(x_train, y_train, patience=10)
        test_score = model.auc(x_test, y_test)
        train_score = model.auc(x_train, y_train)
        scores.append(test_score)
        print('Train %.3f, Test %.3f' % (train_score, test_score))

    return(scores)
