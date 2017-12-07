#!/bin/bash
wget https://www.dropbox.com/s/2prhw0o8e2m8f9k/vectors_1.pkl.syn1neg.npy?dl=1 -O vectors_1.pkl.syn1neg.npy
wget https://www.dropbox.com/s/oqwtlnptp2f7hx2/vectors_1.pkl.wv.syn0.npy?dl=1 -O vectors_1.pkl.wv.syn0.npy
python predict.py $1 $2