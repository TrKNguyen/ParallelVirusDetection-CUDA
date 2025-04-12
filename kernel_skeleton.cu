#include "kseq/kseq.h"
#include "common.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_INTEGRITY 256
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(err);                                                        \
        }                                                                     \
    } while (0)

__global__ void prescan_arbitrary(int *output, int *input, int n, int powerOfTwo)
{
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	
	if (threadID < n) {
		temp[ai + bankOffsetA] = input[ai];
		temp[bi + bankOffsetB] = input[bi];
	}
	else {
		temp[ai + bankOffsetA] = 0;
		temp[bi + bankOffsetB] = 0;
	}
	

	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) {
		temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0; // clear the last element
	}

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[ai] = temp[ai + bankOffsetA];
		output[bi] = temp[bi + bankOffsetB];
	}
}

__global__ void prescan_arbitrary_unoptimized(int *output, int *input, int n, int powerOfTwo) {
	extern __shared__ int temp[];// allocated on invocation
	int threadID = threadIdx.x;

	if (threadID < n) {
		temp[2 * threadID] = input[2 * threadID]; // load input into shared memory
		temp[2 * threadID + 1] = input[2 * threadID + 1];
	}
	else {
		temp[2 * threadID] = 0;
		temp[2 * threadID + 1] = 0;
	}


	int offset = 1;
	for (int d = powerOfTwo >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (threadID == 0) { temp[powerOfTwo - 1] = 0; } // clear the last element

	for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	if (threadID < n) {
		output[2 * threadID] = temp[2 * threadID]; // write results to device memory
		output[2 * threadID + 1] = temp[2 * threadID + 1];
	}
}


__global__ void prescan_large(int *output, int *input, int n, int *sums) {
	extern __shared__ int temp[];

	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;
	
	int ai = threadID;
	int bi = threadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = input[blockOffset + ai];
	temp[bi + bankOffsetB] = input[blockOffset + bi];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) { 
		sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	} 
	
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + ai] = temp[ai + bankOffsetA];
	output[blockOffset + bi] = temp[bi + bankOffsetB];
}

__global__ void prescan_large_unoptimized(int *output, int *input, int n, int *sums) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * n;

	extern __shared__ int temp[];
	temp[2 * threadID] = input[blockOffset + (2 * threadID)];
	temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

	int offset = 1;
	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	__syncthreads();


	if (threadID == 0) {
		sums[blockID] = temp[n - 1];
		temp[n - 1] = 0;
	}

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (threadID < d)
		{
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[blockOffset + (2 * threadID)] = temp[2 * threadID];
	output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
}


__global__ void add(int *output, int length, int *n) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n[blockID];
}

__global__ void add(int *output, int length, int *n1, int *n2) {
	int blockID = blockIdx.x;
	int threadID = threadIdx.x;
	int blockOffset = blockID * length;

	output[blockOffset + threadID] += n1[blockID] + n2[blockID];
}
#define checkCudaError(o, l) _checkCudaError(o, l, __func__)
void _checkCudaError(const char *message, cudaError_t err, const char *caller) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Error in: %s\n", caller);
		fprintf(stderr, message);
		fprintf(stderr, ": %s\n", cudaGetErrorString(err));
		exit(0);
	}
}

void printResult(const char* prefix, int result, long nanoseconds) {
	printf("  ");
	printf(prefix);
	printf(" : %i in %ld ms \n", result, nanoseconds / 1000);
}

void printResult(const char* prefix, int result, float milliseconds) {
	printf("  ");
	printf(prefix);
	printf(" : %i in %f ms \n", result, milliseconds);
}


bool isPowerOfTwo(int x) {
	return x && !(x & (x - 1));
}


int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}

long get_nanos() {
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao);
void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao);

void scanLargeDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int remainder = length % (ELEMENTS_PER_BLOCK);
	if (remainder == 0) {
		scanLargeEvenDeviceArray(d_out, d_in, length, bcao);
	}
	else {
		// perform a large scan on a compatible multiple of elements
		int lengthMultiple = length - remainder;
		scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple, bcao);

		// scan the remaining elements and add the (inclusive) last element of the large scan to this
		int *startOfOutputArray = &(d_out[lengthMultiple]);
		scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder, bcao);

		add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
	}
}

void scanSmallDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	int powerOfTwo = nextPowerOfTwo(length);

	if (bcao) {
		prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
	}
	else {
		prescan_arbitrary_unoptimized<< <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int) >> >(d_out, d_in, length, powerOfTwo);
	}
}

void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length, bool bcao) {
	const int blocks = length / ELEMENTS_PER_BLOCK;
	const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

	int *d_sums, *d_incr;
	cudaMalloc((void **)&d_sums, blocks * sizeof(int));
	cudaMalloc((void **)&d_incr, blocks * sizeof(int));

	if (bcao) {
		prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}
	else {
		prescan_large_unoptimized<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);
	}

	const int sumsArrThreadsNeeded = (blocks + 1) / 2;
	if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
		// perform a large scan on the sums arr
		scanLargeDeviceArray(d_incr, d_sums, blocks, bcao);
	}
	else {
		// only need one block to scan sums arr so can use small scan
		scanSmallDeviceArray(d_incr, d_sums, blocks, bcao);
	}

	add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

	cudaFree(d_sums);
	cudaFree(d_incr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void convertCharToInt(const char* d_sample_qual, int* inArray, int num_elements) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("wtf id %d %d", id, num_elements);
    if (id < num_elements) {
        inArray[id] = d_sample_qual[id] - 33;
        // printf("Thread %d: d_sample_qual[%d] = %c (%d), inArray[%d] = %d\n",
        //     id, id, d_sample_qual[id], d_sample_qual[id], id, inArray[id]);
    } 
}
__global__ void globalPrint(int* arr, int num_elements) {
	for (int i = 0; i < num_elements; i++) {
		printf("%d ", arr[i]);
	}
	printf("\n");
}
void cal_prefix_sum(char* d_sample_qual, int* inArray, int* outArray, int num_elements) {
    // unsigned int mem_size = sizeof(int) * num_elements;
    convertCharToInt<<<(num_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_sample_qual, inArray, num_elements);
	// 	printf("%d\n", num_elements);
	// for (int i = 0; i < num_elements ; i++) {
	// 	printf("%d ", inArray[i]);
	// }
	// printf("\n");
	// for (int i = 0; i < num_elements ; i++) {
	// 	printf("%d ", outArray[i]);
	// }
	// printf("\n");
    // preallocBlockSums(num_elements);
    // prescanArray(outArray, inArray, num_elements);
    scanLargeDeviceArray(outArray, inArray, num_elements, true); 
	// std::cout <<" wtf\n";
	// globalPrint<<<1, 1>>>(inArray, num_elements); 
	// globalPrint<<<1, 1>>>(outArray, num_elements);
	// std::cout <<" wtf\n";
    
}

__global__ void matchKernel(const char* __restrict__ d_samples, 
                            const int* __restrict__ d_sampleLens,
                            const char* __restrict__ d_signatures, 
                            const int* __restrict__ d_sigLens,
                            int* __restrict__ d_match, 
                            int* const* __restrict__ d_prefSum, 
                            int numSig) {
    const int sampleIdx = blockIdx.x;
    const int sigIdx = threadIdx.x;
    if (sigIdx >= numSig) return;
    
    const int offset_sam = d_sampleLens[sampleIdx];
    const int sampleLen = d_sampleLens[sampleIdx + 1] - offset_sam;
    
    const int offset_sig = d_sigLens[sigIdx];
    const int sigLen = d_sigLens[sigIdx + 1] - offset_sig;
    
    const int* prefSum = d_prefSum[sampleIdx];

    int max_score = -1;
    if (sampleLen >= sigLen) {
        for (int pos = 0; pos <= sampleLen - sigLen; pos++) {
            bool ok = true;
            for (int k = 0; k < sigLen; k++) {
                const char a = d_samples[offset_sam + pos + k];
                const char b = d_signatures[offset_sig + k];
                if (a != b && a != 'N' && b != 'N') {
                    ok = false;
                    break;
                }
            }
            if (ok) { 
                int sum = prefSum[pos + sigLen] - prefSum[pos];
                max_score = max(max_score, sum); 
            }
        }
    }
   
    d_match[sampleIdx * numSig + sigIdx] = max_score;
}

__global__ void computeIntegrityKernel(const int* const* __restrict__ d_prefSum,
                                         const int* __restrict__ d_sampleLens,
                                         int* __restrict__ d_integrity,
                                         const int numSam) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < numSam) {
        const int sampleLen = d_sampleLens[id + 1] - d_sampleLens[id];
        d_integrity[id] = d_prefSum[id][sampleLen] % 97;
    }
}

void runMatcher(const std::vector<klibpp::KSeq>& samples,
                const std::vector<klibpp::KSeq>& signatures,
                std::vector<MatchResult>& matches) {
    int numSam = samples.size();
    int numSig = signatures.size();

    // Allocate samples
    int* h_sampleLens = new int[numSam + 1];
    h_sampleLens[0] = 0;
    size_t total_sam = 0;
    for (int i = 0; i < numSam; i++) {
        size_t size = samples[i].seq.size();
        total_sam += size;
        h_sampleLens[i + 1] = h_sampleLens[i] + size;
    }
    
    char* h_samples = new char[total_sam];
    for (int i = 0; i < numSam; i++) {
        for (size_t j = 0; j < samples[i].seq.size(); j++) {
            h_samples[h_sampleLens[i] + j] = samples[i].seq[j];
        }
    }

    int** h_prefSum = new int*[numSam];
    for (int i = 0; i < numSam; i++) {
        size_t qualSize = samples[i].qual.size() + 1;
        char* d_qual;
        CUDA_CHECK(cudaMalloc((void**)&d_qual, qualSize));
        CUDA_CHECK(cudaMemcpy(d_qual, samples[i].qual.c_str(), qualSize, cudaMemcpyHostToDevice));
        
        int* in_array; 
        int* out_array;
        CUDA_CHECK(cudaMalloc((void**)&in_array, qualSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&out_array, qualSize * sizeof(int)));
        cal_prefix_sum(d_qual, in_array, out_array, qualSize);
        h_prefSum[i] = out_array;
 
        CUDA_CHECK(cudaFree(d_qual));
        CUDA_CHECK(cudaFree(in_array));
    }
    
    // Allocate samples
    char* d_samples;
    CUDA_CHECK(cudaMalloc((void**)&d_samples, total_sam));
    CUDA_CHECK(cudaMemcpy(d_samples, h_samples, total_sam, cudaMemcpyHostToDevice));
    
    int* d_sampleLens;
    CUDA_CHECK(cudaMalloc((void**)&d_sampleLens, (numSam + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sampleLens, h_sampleLens, (numSam + 1) * sizeof(int), cudaMemcpyHostToDevice));
    
    // Allocate prefix sum
    int** d_prefSum;
    CUDA_CHECK(cudaMalloc((void**)&d_prefSum, numSam * sizeof(int*)));
    CUDA_CHECK(cudaMemcpy(d_prefSum, h_prefSum, numSam * sizeof(int*), cudaMemcpyHostToDevice));

    // Allocate for signatures
    int* h_sigLens = new int[numSig + 1];
    h_sigLens[0] = 0;
    size_t total_sig = 0;
    for (int i = 0; i < numSig; i++) {
        size_t size = signatures[i].seq.size();
        total_sig += size;
        h_sigLens[i + 1] = h_sigLens[i] + size;
    }
    
    char* h_sigs = new char[total_sig];
    for (int i = 0; i < numSig; i++) {
        for (size_t j = 0; j < signatures[i].seq.size(); j++) {
            h_sigs[h_sigLens[i] + j] = signatures[i].seq[j];
        }
    }
    
    char* d_signatures;
    CUDA_CHECK(cudaMalloc((void**)&d_signatures, total_sig));
    CUDA_CHECK(cudaMemcpy(d_signatures, h_sigs, total_sig, cudaMemcpyHostToDevice));
    
    int* d_sigLens;
    CUDA_CHECK(cudaMalloc((void**)&d_sigLens, (numSig + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_sigLens, h_sigLens, (numSig + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate output array for match scores.
    int* d_match;
    CUDA_CHECK(cudaMalloc((void**)&d_match, numSam * numSig * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_match, 0xFF, numSam * numSig * sizeof(int)));

    // Run find match
    dim3 gridDim(numSam, 1, 1);
    dim3 blockDim(numSig, 1, 1);
    matchKernel<<<gridDim, blockDim>>>(d_samples, d_sampleLens, d_signatures, d_sigLens, d_match, d_prefSum, numSig);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy match scores back to host.
    int* h_match = new int[numSam * numSig];
    CUDA_CHECK(cudaMemcpy(h_match, d_match, numSam * numSig * sizeof(int), cudaMemcpyDeviceToHost));

	// Calculate integrity 
    int* d_integrity;
    CUDA_CHECK(cudaMalloc((void**)&d_integrity, numSam * sizeof(int)));
    dim3 gridDimIntegrity((numSam + BLOCK_INTEGRITY - 1) / BLOCK_INTEGRITY, 1, 1);
    dim3 blockDimIntegrity(BLOCK_INTEGRITY, 1, 1);
    computeIntegrityKernel<<<gridDimIntegrity, blockDimIntegrity>>>(d_prefSum, d_sampleLens, d_integrity, numSam);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    int* h_integrity = new int[numSam];
    CUDA_CHECK(cudaMemcpy(h_integrity, d_integrity, numSam * sizeof(int), cudaMemcpyDeviceToHost));

    // Build match result
    for (int i = 0; i < numSam; i++) {
        for (int j = 0; j < numSig; j++) {
            int score = h_match[i * numSig + j];
            if (score != -1) {
                MatchResult res;
                res.sample_name = samples[i].name;
                res.signature_name = signatures[j].name;
                int sigLen = h_sigLens[j + 1] - h_sigLens[j];
                res.match_score = static_cast<double>(score) / sigLen;
                res.integrity_hash = h_integrity[i];
                matches.push_back(res);
            }
        }
    }

    // Free host memory.
    delete[] h_samples;
    delete[] h_sampleLens;
    delete[] h_sigLens;
    delete[] h_sigs;
    delete[] h_match;
    delete[] h_integrity;

    // Free device memory.
    CUDA_CHECK(cudaFree(d_samples));
    CUDA_CHECK(cudaFree(d_sampleLens));
    CUDA_CHECK(cudaFree(d_signatures));
    CUDA_CHECK(cudaFree(d_sigLens));
    CUDA_CHECK(cudaFree(d_match));
    CUDA_CHECK(cudaFree(d_integrity));
    CUDA_CHECK(cudaFree(d_prefSum));

    // Also free device prefix sum arrays.
    for (int i = 0; i < numSam; i++) {
        CUDA_CHECK(cudaFree(h_prefSum[i]));
    }
	delete[] h_prefSum;
}