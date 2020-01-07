# FAISS Experiments

This is a playground for FAISS KNN/ANN.
## Configure

You need first install your FAISS properly and modify the makefile.inc first. And then make all to create examples for testing.

```bash
make all
```

## Experiments

### Street View HOuse Numbers (SVHN) dataset

### NVPROF Command


```bash
nvprof --log-file prof.csv --csv ./Multiple-GPUs # to csv
nvprof --export-profile a.nvvp ./Multiple-GPUs   # to nvvp visualization
```

