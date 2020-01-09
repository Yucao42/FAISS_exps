#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>

#include "Config.h"


int main() {
    // Basic parameters
    size_t d = 128;                              // feature dimension
    const int nlist =  100000;                   // number of centroids in coarse quantizer
    const int m = 8;                             // sub-quantizer
    const int k = 10;                            // k-NN k.

    // Load sift 1M Data
    // Training data
    size_t nt;
    float *xt = fvecs_read("sift/sift_learn.fvecs", &d, &nt);

    // Database data
    size_t nb, d2;
    float *xb = fvecs_read("sift/sift_base.fvecs", &d2, &nb);
    assert(d == d2 || !"dataset does not have same dimension as train set");

    // Query data
    size_t d3, nq;
    float *xq = fvecs_read("sift/sift_query.fvecs", &d3, &nq);
    assert(d == d3 || !"query does not have same dimension as train set");

    // If predefined sizes
    CHECK_REPLACE(nq, nq_)
    CHECK_REPLACE(nb, nb_)
    printf("number of base vectors %d \n", nb);

    int ngpus = faiss::gpu::getNumDevices();
    printf("Number of GPUs: %d\n", ngpus);

    std::vector<faiss::gpu::GpuResources*> res;
    std::vector<int> devs;
    for(int i = 0; i < ngpus; i++) {
        res.push_back(new faiss::gpu::StandardGpuResources);
        if(i == 1){
            devs.push_back(i);
        }
    }

    // If sharded or replicated in multiple GPUs.
    faiss::gpu::GpuMultipleClonerOptions option;
    option.shard = true;


#if GPU_KNN_TEST >0
    faiss::IndexFlatL2 cpu_index(d);

    printf("migrating index from cpu to gpu\n");
    start = std::chrono::steady_clock::now();
    faiss::Index *gpu_index;
    if (devs.size() > 1){
        gpu_index =faiss::gpu::index_cpu_to_gpu_multiple(
                      res,
                      devs,
                      &cpu_index,
                      &option
                  );}
    else{
        gpu_index =faiss::gpu::index_cpu_to_gpu(
                      res[0],
                      1,
                      &cpu_index
                  );
    }
    end = std::chrono::steady_clock::now();
    std::cout << "[Migrating time] "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
         << " us" << std::endl;

    printf("is_trained = %s\n", gpu_index->is_trained ? "true" : "false");
    start = std::chrono::steady_clock::now();
    gpu_index->add(nb, xb);  // add vectors to the index
    end = std::chrono::steady_clock::now();
    printf("ntotal = %ld\n", gpu_index->ntotal);
    std::cout << "[ADD TIME] [GPU BF] add " << nb << " records in microseconds : "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
         << " us" << std::endl;

    if (do_sanity_check)
    {       // sanity check: search 5 first vectors of xb
        long *I = new long[k * sanity_query_number];
        float *D = new float[k * sanity_query_number];

        start = std::chrono::steady_clock::now();
        gpu_index->search(sanity_query_number, xb, k, D, I);
        end = std::chrono::steady_clock::now();
        std::cout << "[SEARCH TIME] [GPU BF] search " << sanity_query_number << " records in microseconds : "
             << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
             << " us" << std::endl;

        // print results
        printf("I=\n");
        for(int i = 0; i < sanity_query_number; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < sanity_query_number; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        start = std::chrono::steady_clock::now();
        gpu_index->search(nq, xq, k, D, I);
        end = std::chrono::steady_clock::now();
        std::cout << "[SEARCH TIME] [GPU BF] search " << nq << " records in microseconds : "
             << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
             << " us" << std::endl;

        // Test query time for query number from 0 to test_max_number
        TEST_TIME_FUNC(gpu_index->search(i, xq, k, D, I), test_max_number, "GPU_BF")

        // print results
        if(do_print_results)
        {
          printf("I (5 first results)=\n");
          for(int i = 0; i < 5; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }

          printf("I (5 last results)=\n");
          for(int i = nq - 5; i < nq; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }
        }

        delete [] I;
        delete [] D;
    }

    delete gpu_index;

#endif

// CPU
#if CPU_KNN_TEST > 0
    faiss::IndexFlatL2 index(d);           // call constructor
    printf("is_trained = %s\n", index.is_trained ? "true" : "false");

    start = std::chrono::steady_clock::now();
    index.add(nb, xb);                     // add vectors to the index
    end = std::chrono::steady_clock::now();
    std::cout << "[ADD TIME] [CPU BF] add " << nb << " records in microseconds : "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
         << " us" << std::endl;
    printf("ntotal = %ld\n", index.ntotal);


    if (do_sanity_check)
    {       // sanity check: search 5 first vectors of xb
        long *I = new long[k * sanity_query_number];
        float *D = new float[k * sanity_query_number];

        auto start = std::chrono::steady_clock::now();
        index.search(sanity_query_number, xb, k, D, I);
        auto end = std::chrono::steady_clock::now();
        std::cout << "[SEARCH TIME] [CPU BF] search " << sanity_query_number << " records in microseconds : "
             << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
             << " us" << std::endl;

        // print results
        printf("I=\n");
        for(int i = 0; i < sanity_query_number; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < sanity_query_number; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }


    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        auto start = std::chrono::steady_clock::now();
        index.search(nq, xq, k, D, I);
        auto end = std::chrono::steady_clock::now();
        std::cout << "[SEARCH TIME] [CPU BF]Search time for queries " << nq << " records in microseconds : "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
         << " us" << std::endl;

        // Test query time for query number from 0 to test_max_number
        TEST_TIME_FUNC(index.search(i, xq, k, D, I), test_max_number, "CPU_BF")

        // print results
        if(do_print_results)
        {
          printf("I (5 first results)=\n");
          for(int i = 0; i < 5; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }

          printf("I (5 last results)=\n");
          for(int i = nq - 5; i < nq; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }
        }

        delete [] I;
        delete [] D;
    }
#endif


#if CPU_ANN_TEST > 0
    {
      faiss::IndexFlatL2 quantizer(d);       // the other index
      faiss::IndexIVFPQ index(&quantizer, d, nlist, m, 8);
      // here we specify METRIC_L2, by default it performs inner-product search
      start = std::chrono::steady_clock::now();
      index.train(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[TRAIN TIME] [CPU ANN] " << nb << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;


      start = std::chrono::steady_clock::now();
      index.add(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[ADD TIME] [CPU ANN] " << 1000 << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;
      
      {       
          long *I = new long[k * nq];
          float *D = new float[k * nq];

          index.nprobe = 10;
          start = std::chrono::steady_clock::now();
          index.search(nq, xq, k, D, I);
          end = std::chrono::steady_clock::now();
          std::cout << "[SEARCH TIME] [CPU ANN] " << 1000 << " records in microseconds : "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
           << " us" << std::endl;

          printf("I=\n");
          for(int i = nq - 5; i < nq; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }

          delete [] I;
          delete [] D;
      }
    }
#endif

#if GPU_ANN_TEST > 0
    {
      faiss::IndexFlatL2 quantizer(d);       // the other index
      faiss::IndexIVFPQ index_cpu(&quantizer, d, nlist, m, 8);
      // here we specify METRIC_L2, by default it performs inner-product search
      index_cpu.nprobe = 10;
      index_cpu.verbose = true;
      std::cout << "Before" << nb << " CASTing" << std::endl; 
      faiss::Index *gpu_index =
          faiss::gpu::index_cpu_to_gpu_multiple(
              res,
              devs,
              &index_cpu,
              &option
          );
      std::cout << "After" << nb << " CASTing" << std::endl; 
      
      start = std::chrono::steady_clock::now();
      gpu_index->train(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[TRAIN TIME] [GPU ANN] " << nb << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;
      //faiss::gpu::GpuIndexIVFFlat index_ivf(&res, d, nlist, faiss::METRIC_L2);

      start = std::chrono::steady_clock::now();
      gpu_index->add(nb, xb);
      end = std::chrono::steady_clock::now();
      std::cout << "[ADD TIME] [CPU ANN] " << 1000 << " records in microseconds : "
       << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
       << " us" << std::endl;

      {       // search xq
          long *I = new long[k * nq];
          float *D = new float[k * nq];

          start = std::chrono::steady_clock::now();
          gpu_index->search(nq, xq, k, D, I);
          end = std::chrono::steady_clock::now();
          std::cout << "[SEARCH TIME] [CPU ANN] " << 1000 << " records in microseconds : "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
           << " us" << std::endl;

          printf("I=\n");
          for(int i = nq - 5; i < nq; i++) {
              for(int j = 0; j < k; j++)
                  printf("%5ld ", I[i * k + j]);
              printf("\n");
          }

          delete [] I;
          delete [] D;
      }
    }
#endif
    delete [] xb;
    delete [] xq;
    for(int i = 0; i < ngpus; i++) {
        delete res[i];
    }

    return 0;
}
