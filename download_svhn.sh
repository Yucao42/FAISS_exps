#!/bin/bash
here=$(dirname "$0")
[[ "$here" = /*  ]] || here="$PWD/$here"
echo $here

DATA_DIR=/data/yc3390/faiss_data
UNAME=$(uname)
if [ "$UNAME" == "Darwin" ];then
    DATA_DIR=$HOME/vas/data;
fi

mkdir -p $DATA_DIR
ln -s $DATA_DIR data
mkdir -p data/SVHN && cd data/SVHN
wget http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget http://ufldl.stanford.edu/housenumbers/extra_32x32.mat
cd $here
