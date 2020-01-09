# Parse the config.h file
TEST=Time
if [[ $(cat Config.h | grep CPU_KNN | grep 1) ]]; then
    TEST=${TEST}_CPU.KNN
fi
if [[ $(cat Config.h | grep GPU_KNN | grep 1) ]]; then
    TEST=${TEST}_GPU.KNN
fi

NUMBER_QUERY=$(cat Config.h | grep nq_ | tr -dc '0-9')
NUMBER_BASE=$(cat Config.h | grep nb_ | tr -dc '0-9')
if [ "$NUMBER_QUERY" = "0" ]; then
    NUMBER_QUERY=10K # Default query number
fi
if [ "$NUMBER_BASE" = "0" ]; then
    NUMBER_BASE=1M # Default training number
fi
TEST=${TEST}_BaseNumber.${NUMBER_BASE}_QueryNumber.${NUMBER_QUERY}

if [[ $(cat Config.h | grep undef | grep //) ]]; then
    NUMBER=$(cat Config.h | grep test_max_number | tr -dc '0-9')
    TEST=${TEST}_QueryNumberRangeTest.${NUMBER} 
    TEST_TIME=1
fi

echo Test name is ${TEST}

# NVVP settings
CSV=1
NVVP=1
LOGFILE=0
NVVP_PROFILE=0

# Directories
LOG_DIR=logs/nvprofs
CSV_DIR=${LOG_DIR}/csv
NVVP_DIR=${LOG_DIR}/nvvp

mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}
mkdir -p ${NVVP_DIR}
make clean && make all

# NVVP for visualization
if [ "$TEST_TIME" = "1" ]; then
    ./Benchmarker
    exit
fi

# NVVP for visualization
if [ "$NVVP" = "1" ]; then
    nvprof --export-profile ${NVVP_DIR}/${TEST}.nvvp -f ./Benchmarker
fi

# CSV file for reading
if [ "$CSV" = "1" ]; then
    nvprof --log-file ${CSV_DIR}/${TEST}.csv --csv ./Benchmarker
fi
