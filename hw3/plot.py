from keras.utils.vis_utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# model = load_model('data/DNN_model.h5')
# plot_model(model, to_file='result/DNN_model.png')
# model.summary()
# model = load_model('data/model9.h5')
# plot_model(model, to_file='result/CNN_model.png')
# model.summary()

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


model = load_model('data/model9.h5')
model.load_weights('fuckme.hdf5')
X_train = np.load('X_valid.npy')
pred = model.predict(X_train)
pred = pred.argmax(axis=-1)
data = pd.read_csv('data/train.csv')
y_train = data.iloc[:, 0].values.astype(int)
y_train = np.load('y_valid.npy')
y_train = y_train.argmax(axis=-1)

conf_mat = confusion_matrix(y_train, pred)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"])
plt.savefig('result/conf_mat.png')
