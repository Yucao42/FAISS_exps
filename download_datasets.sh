#!/bin/bash
here=$(dirname "$0")
[[ "$here" = /*  ]] || here="$PWD/$here"
echo $here

# Initial data folder
DATA_DIR=/data/yc3390/faiss_data
UNAME=$(uname)
if [ "$UNAME" == "Darwin" ];then
    DATA_DIR=$HOME/vas/data;
fi
mkdir -p $DATA_DIR
if [ ! -L data ]; then
    ln -s $DATA_DIR data
fi

DATASETS=$1
if [ -z "$DATASETS" ]; then
    DATASETS=SVHN,phototour
fi

IFS=","
CHOICES=("SVHN" "phototour")

HTTP_SVHN=http://ufldl.stanford.edu/housenumbers/
HTTP_phototour=http://phototour.cs.washington.edu/patches/

FILES_SVHN=(train_32x32.mat,test_32x32.mat,extra_32x32.mat)
FILES_phototour=(trevi.zip,halfdome.zip,notredame.zip)

# Prepare data
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'
for DATASET in $DATASETS;
do
    if [[ ! " ${CHOICES[@]} " =~ " $DATASET " ]]; then
        echo ${DATASET} is NOT in the supported list
        echo Supported datasets are:
        echo ${CHOICES[*]}
        echo -e ${RED}Exit with an unsupported dataset ${DATASET} ${NC}
        tput sgr0
        exit
    fi
    echo "Preparing dataset $DATASET";
    mkdir -p data/${DATASET} && cd data/${DATASET}
    HTTP=HTTP_${DATASET}
    FILES=FILES_${DATASET}
    
    for FILE in ${!FILES};
    do
        if [ ! -f ${FILE} ]; then
            echo Downloading ${!HTTP}${FILE}
            wget ${!HTTP}${FILE}
        else
            echo ${FILE} has been downloaded.
        fi
    done
    echo -e ${GREEN}Done downloading $DATASET 
    tput sgr0
    cd $here
done
unset IFS
