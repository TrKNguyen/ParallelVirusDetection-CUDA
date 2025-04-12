# Parallel Virus Detection - CUDA

**1. Overview:**
This project implements a parallel virus signature scanning algorithm using CUDA for efficient virus detection in biological DNA samples. The approach applies a brute-force sliding window technique to compare sample DNA sequences against virus signatures, utilizing GPU parallelization for enhanced performance.

**2. Algorithm:**
The algorithm uses a sliding window approach to find matches between sample and virus signatures, allowing for wildcard matches ('N'). It computes match confidence based on Phred scores and calculates an integrity hash to verify data consistency. A parallel prefix-sum (scan) algorithm optimizes performance by precomputing the qualification scores, reducing unnecessary recalculations.

**3. Parallelization Strategy:**
- The algorithm leverages CUDA for parallel execution.
- Each sample is assigned a separate block in the grid, and each thread within a block processes one virus signature.
- Memory handling is optimized with shared memory to store partial sums during prefix-sum operations.

**4. Optimization Efforts:**
- **First Optimization:** Precomputing the prefix sum array(using scan algorithm to compute it in O(logn)) for sample qualification scores eliminates redundant calculations.
- **Second Optimization:** Flattening 2D arrays into 1D arrays improves memory access patterns, enhancing cache locality and performance.

**5. Files Included:**
- **`kernel_skeleton.cu`:** Contains the core CUDA implementation for virus signature matching.
- **`Report.pdf`:** Detailed implementation report and performance analysis.
- **`common.h`/`common.cc`:** Entry point for handling input/output and performance timing.
- **`Makefile`:** Used to compile the CUDA project.
- **`gen_sample.cc`/`gen_sig.cc`:** Helpers for generating sample and signature input files.
