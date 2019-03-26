from numpy import vstack
from sklearn import preprocessing
from constants import SCALE_LIMITS

def scale_data_single(X_train):
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(SCALE_LIMITS['low'], SCALE_LIMITS['upp'])).fit(X_train)
    X_train_scale = minmax_scaler.transform(X_train)
    return X_train_scale

def scale_data_multiple(X_train, x_test):
    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(SCALE_LIMITS['low'], SCALE_LIMITS['upp'])).fit(vstack((X_train, x_test)))
    X_train_scale = minmax_scaler.transform(X_train)
    x_test_scale = minmax_scaler.transform([x_test])
    return X_train_scale, x_test_scale
