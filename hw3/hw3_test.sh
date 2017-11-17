#!/bin/bash
wget https://www.dropbox.com/s/y333yyi57b1w5ue/model7.h5?dl=1 -O model7.h5
wget https://www.dropbox.com/s/lfz0lx64sb4t6kg/model8.h5?dl=1 -O model8.h5
wget https://www.dropbox.com/s/27hcf4zkkkj1j0s/model9.h5?dl=1 -O model9.h5
wget https://www.dropbox.com/s/h3ef84c07ze9yxf/modelA.h5?dl=1 -O modelA.h5
wget https://www.dropbox.com/s/0v2ogb4i4lisj40/modelB.h5?dl=1 -O modelB.h5
wget https://www.dropbox.com/s/u7rdtukfu5v37ke/w7.hdf5?dl=1 -O w7.hdf5
wget https://www.dropbox.com/s/dcjn6j8ter8p27u/w8.hdf5?dl=1 -O w8.hdf5
wget https://www.dropbox.com/s/4ac8quwn0acyu04/w9.hdf5?dl=1 -O w9.hdf5
wget https://www.dropbox.com/s/vcjrs0ouc8oah36/wA.hdf5?dl=1 -O wA.hdf5
wget https://www.dropbox.com/s/3a7tx0rbdpzmree/wB.hdf5?dl=1 -O wB.hdf5
python predict.py $1 $2