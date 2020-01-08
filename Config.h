#ifndef CONFIG_H
#define CONFIG_H

#define CPU_KNN_TEST 1     
#define GPU_KNN_TEST 1
#define CPU_ANN_TEST 0     
#define GPU_ANN_TEST 0

// if multiple time is tested
#define TEST_TIME

// Sanity check using samples
const int sanity_query_number = 1;

// Test 0 to test_maxtime samples
const int test_max_number = 1000;

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

#define PRINT_TIME_FUNC(EXPR, method_, job_)                                           \
    start = std::chrono::steady_clock::now();                                          \
    EXPR;                                                                              \
    end = std::chrono::steady_clock::now();                                            \
    std::cout << "[TIME] for job_ using method_  " << " in microseconds : "            \
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() \
         << " us" << std::endl;                                                        \

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