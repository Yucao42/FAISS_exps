#ifndef CONFIG_H
#define CONFIG_H

#define CPU_KNN_TEST 1
#define GPU_KNN_TEST 0
#define CPU_ANN_TEST 0
#define GPU_ANN_TEST 0

// if the data is initialized from random
#define RANDOM_INIT 1

// if multiple time is tested
#define TEST_TIME
// #undef TEST_TIME

// Data dimension
const int dimension_ = 4;

// Do sanity check or not
const bool do_sanity_check = false;

// Print results or not
const bool do_print_results = false;

// Sanity check using samples
const int sanity_query_number = 1;

// Test 0 to test_maxtime samples
const int test_max_number = 1000;

// Explicit set number of training, default 0 will be ignored
const size_t nt_ = 10000;

// Explicit set number of query, default 0 will be ignored
const size_t nq_ = 10000;

// Explicit set number of base vectors added to DB, default 0 will be ignored
const size_t nb_ = 256;

// Number of centroids for K-means
const int nlist = 256;         

// sub-quantizer
const int m = 8;                             

// k-NN k
const int k = 10;

// GPU device number to be used
const int gpu_devno_ = 0;

// Timers
auto start = std::chrono::steady_clock::now();
auto end = std::chrono::steady_clock::now();

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef TEST_TIME
#define TEST_TIME_FUNC(EXPR, MAX_TIME, TEST_NAME)                                          \
  {                                                                                        \
    for(int i = 0; i < MAX_TIME; i ++){                                                    \
      start = std::chrono::steady_clock::now();                                            \
      EXPR;                                                                                \
      end = std::chrono::steady_clock::now();                                              \
      std::cout << "[SEARCH TIME] "<< TEST_NAME << " " << i << " records in us: "          \
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()   \
           << " us" << std::endl;                                                          \
    }                                                                                      \
  }                                                                                        
#else
#define TEST_TIME_FUNC(EXPR, MAX_TIME, TEST_NAME) {}
#endif

#define PRINT_TIME_FUNC(EXPR, method_, job_)                                               \
    start = std::chrono::steady_clock::now();                                              \
    EXPR;                                                                                  \
    end = std::chrono::steady_clock::now();                                                \
    std::cout << "[TIME] for job_ using method_  " << " in microseconds : "                \
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()     \
         << " us" << std::endl;                                                            \

#define CHECK_REPLACE(X, X_)                                                               \
  {                                                                                        \
    if(X_){                                                                                \
      X = MIN(X, X_);                                                                      \
    }                                                                                      \
  }                                                                                        \

float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}

#endif
