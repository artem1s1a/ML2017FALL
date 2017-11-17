import os
from keras.models import load_model
from termcolor import colored, cprint
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


def deprocessimage(x):
    """
    Hint: Normalize and Clip
    """
    return x


base_dir = os.path.dirname(os.path.dirname(os.path.realpath('map.py')))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir, 'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)


def main():
    emotion_classifier = load_model('data/model9.h5')
    emotion_classifier.load_weights('data/weights9.69-0.65.hdf5')
    print(colored("Loaded model from {}".format('data/X_train.npy'), 'yellow', attrs=['bold']))

    X_train = np.load('data/X_train.npy')
    X_train /= 255
    input_img = emotion_classifier.input
    img_ids = [87]

    for idx in img_ids:
        img = X_train[idx].reshape((1, 48, 48, 1))
        val_proba = emotion_classifier.predict(img)
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])
        """
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        """
        layer_dict = dict([(layer.name, layer) for layer in emotion_classifier.layers])
        target_layer = layer_dict['dense_4']
        [heatmap] = fn([img, target_layer])
        heatmap = heatmap.reshape((48, 48))
        heatmap = np.abs(heatmap)
        heatmap = heatmap / np.max(heatmap)
        thres = 0.1
        see = X_train[idx].reshape(48, 48)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('see.png')

        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(cmap_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}_1.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(partial_see_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}_2.png'.format(idx)), dpi=100)


if __name__ == "__main__":
    main()
