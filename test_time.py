# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python2

from __future__ import print_function
import os
import time
import numpy as np
from IPython import embed

TEST_NUMBER_MAX = 10000
random = False
try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    graphical_output = True
except ImportError:
    graphical_output = False

import faiss

#################################################################
# Small I/O functions
#################################################################

def ivecs_read(fname):
    f = open(fname)
    d, = np.fromfile(f, count = 1, dtype = 'int32')
    sz = os.stat(fname).st_size
    assert sz % (4 * (d + 1)) == 0
    n = sz // (4 * (d + 1))
    f.seek(0)
    a = np.fromfile(f, count = n * (d +1), dtype = 'int32').reshape(n, d + 1)
    return a[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def plot_OperatingPoints(ops, nq, **kwargs):
    ops = ops.optimal_pts
    n = ops.size() * 2 - 1
    print([ops.at( i      // 2).perf for i in range(n)],
                [ops.at((i + 1) // 2).t / nq * 1000 for i in range(n)])
    pyplot.plot([ops.at( i      // 2).perf for i in range(n)],
                [ops.at((i + 1) // 2).t / nq * 1000 for i in range(n)],
                **kwargs)


#################################################################
# prepare common data for all indexes
#################################################################



t0 = time.time()

print("load data")

xt = fvecs_read("sift/sift_learn.fvecs")
xb = fvecs_read("sift/sift_base.fvecs")
xq = fvecs_read("sift/sift_query.fvecs")
print(xq[0])
if random:
    xt = np.random.random(xt.shape)
    xb = np.random.random(xb.shape)
    xq = np.random.random(xq.shape)

d = xt.shape[1]

print("load GT")

gt = ivecs_read("sift/sift_groundtruth.ivecs")
gt = gt.astype('int64')
k = gt.shape[1]
k = 3

print("prepare criterion")

# criterion = 1-recall at 1
crit = faiss.OneRecallAtRCriterion(xq.shape[0], 100)
crit.set_groundtruth(None, gt)
crit.nnn = k

# indexes that are useful when there is no limitation on memory usage
unlimited_mem_keys = [
    "IMI2x10","Flat", "IMI2x11,Flat",
    "IVF4096,Flat", "IVF16384,Flat",
    "PCA64,IMI2x10,Flat"]

# memory limited to 16 bytes / vector
keys_mem_16 = [
    'IMI2x10,PQ16', 'IVF4096,PQ16',
    'IMI2x10,PQ8+8', 'OPQ16_64,IMI2x10,PQ16'
    ]

# limited to 32 bytes / vector
keys_mem_32 = [
    'IMI2x10,PQ32', 'IVF4096,PQ32', 'IVF16384,PQ32',
    'IMI2x10,PQ16+16',
    'OPQ32,IVF4096,PQ32', 'IVF4096,PQ16+16', 'OPQ16,IMI2x10,PQ16+16'
    ]

# indexes that can run on the GPU
keys_gpu = [
    "PCA64,IVF4096,Flat",
    "PCA64,Flat", "Flat", "IVF4096,Flat", "IVF16384,Flat",
    "IVF4096,PQ32"]


keys_to_test = unlimited_mem_keys
keys_to_test = keys_gpu
keys_to_test = ["PQ32", "IVF4096,Flat", "IVF2048,PQ32", "IVF4096,PQ32"]
keys_to_test = ["Flat"]
use_gpu = True
# use_gpu = False


if use_gpu:
    # if this fails, it means that the GPU version was not comp
    assert faiss.StandardGpuResources, \
        "FAISS was not compiled with GPU support, or loading _swigfaiss_gpu.so failed"
    res = faiss.StandardGpuResources()
    dev_no = 1

training_time_cpu = []
indexing_time_cpu = []
training_time_gpu = []
indexing_time_gpu = []
search_time = [[] for _ in range(2)]
db_size = 100000
# xb = xb[:db_size]
print(xb.shape)


for use_gpu in range(2):
    # remember results from other index types
    # use_gpu=1
    op_per_key = []
    
    
    # keep track of optimal operating points seen so far
    op = faiss.OperatingPoints()
    xq_ = xq[:1, ...]
    for index_key in keys_to_test:
        for i in range(2, TEST_NUMBER_MAX):
            print("============ key", index_key)
    
            # make the index described by the key
            index = faiss.index_factory(d, index_key)
            if use_gpu:
                # transfer to GPU (may be partial)
                index = faiss.index_cpu_to_gpu(res, dev_no, index)
                params = faiss.GpuParameterSpace()
            else:
                params = faiss.ParameterSpace()
    
            t1 = time.time()
            xb_ = xb[:i]
            index.add(xb_)
            t1 = time.time()
            D, I = index.search(xq_, k)
            t = (time.time() - t1) * 1000
            print(i, "Time elapse {:.2f} ms".format(t))
            search_time[use_gpu].append(t)
    


# Draw time distribution
print("searching time cpu: \n", search_time[0], '\n')
print("searching time gpu: \n", search_time[1], '\n')
import numpy as np
import matplotlib.pyplot as plt

log_scale = False
platform = ['CPU', 'GPU']
time = list(range(2, TEST_NUMBER_MAX))
for use_gpu in range(2):
    # use_gpu=1
    name = platform[use_gpu] + ' KNN Search Time Test in SIFT 1M'
    plt.clf()
    plt.plot(time, search_time[use_gpu], label= platform[use_gpu] + ' KNN search')
    plt.legend(loc='upper left')
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
    # Add title and x, y labels
    plt.title(name, fontsize=14, fontweight='bold')
    plt.xlabel("Query Batch number (q)")
    plt.ylabel(f"Time in milleseconds")
    plt.savefig(f'{name}.jpg'.replace(" ", '_'), dpi=400)

np.save(f'result_{TEST_NUMBER_MAX}.npy', np.asarray(search_time))
np.save(f'time_{TEST_NUMBER_MAX}.npy', np.asarray(time))
