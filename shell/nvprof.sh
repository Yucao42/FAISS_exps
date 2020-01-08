# Parse the config.h file
TEST=Time
if [[ $(cat Config.h | grep CPU_KNN | grep 1) ]]; then
    TEST=${TEST}_CPU.KNN
fi

NUMBER_QUERY=$(cat Config.h | grep nq_ | tr -dc '0-9')
NUMBER_BASE=$(cat Config.h | grep nb_ | tr -dc '0-9')
TEST=${TEST}_QueryNumberRangeTest.${NUMBER} 
if [ "$NUMBER_QUERY" = "0" ]; then
    NUMBER_QUERY=10K # Default query number
fi
if [ "$NUMBER_BASE" = "0" ]; then
    NUMBER_BASE=1M # Default training number
fi
TEST=${TEST}_BaseNumber.${NUMBER_BASE}_QueryNumber.${NUMBER_QUERY}

if [[ $(cat Config.h | grep GPU_KNN | grep 1) ]]; then
    TEST=${TEST}_GPU.KNN
fi
if [[ $(cat Config.h | grep undef | grep //) ]]; then
    NUMBER=$(cat Config.h | grep test_max_number | tr -dc '0-9')
    TEST=${TEST}_QueryNumberRangeTest.${NUMBER} 
fi

echo Test name is ${TEST}

# NVVP settings
CSV=0
LOGFILE=0
NVVP_PROFILE=results.nvvp
LOG_DIR=logs/nvprofs

mkdir -p LOG_DIR
make clean && make all

nvprof --log-file ${LOG_DIR}/${TEST}.csv --csv ./Benchmarker # to csv
