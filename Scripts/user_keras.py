from sklearn import preprocessing

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import regularizers

# standardization
SS = preprocessing.StandardScaler()
X_all_std = SS.fit_transform(X_all)
X_std = X_all_std[:n_train, :]

X_train, X_val, y_train, y_val = train_test_split(X_std, y, test_size=0.1, random_state=0)
label_train = np_utils.to_categorical(y_train)
label_val = np_utils.to_categorical(y_val)

# a linear stack of layers
model = Sequential()

model.add(Dense(64, input_dim=X.shape[1]))
model.add(Activation('relu'))
#model.add(Dropout(0.05))

model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.05))

model.add(Dense(num_class))
model.add(Activation('softmax'))

# multiclass logloss
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

# fit
# a typical minibatch size is 256
# shuffle the samples at each epoch
model.fit(X_train, label_train, batch_size=128, nb_epoch=20, validation_data=(X_val, label_val),
	shuffle=True, show_accuracy=True, verbose=2)

preds = model.predict_proba(X_val, verbose=0)
ndcg5(preds, xg_val)

X_test_std = X_all_std[n_train:, :]
y_pred_nn = model.predict_proba(X_test_std, verbose=0)
