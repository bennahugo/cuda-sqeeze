/*
    Predictive 32-bit IEEE 754 floating point data compressor
    Copyright (C) 2013  benjamin bennahugo@aol.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "gpuCode.h"

uint32_t * _gpuCompressorIV = NULL;
uint64_t _gpuCompressorIVLength = -1;
uint32_t _gpuAccumCompressedDataSize = 0;
uint32_t * _gpuDecompressorIV = NULL;
uint64_t _gpuDecompressorIVLength = -1;
double _gpuCompressorAccumulatedTime = 0;
double _gpuDecompressorAccumulatedTime = 0;
uint32_t _gpuAccumDecompressedDataSize = 0;
const uint8_t gpuStorageIndiceCapacity = 8*sizeof(uint32_t);
const uint8_t gpuBitCountForRepresentation = 2;
uint32_t gpuBlockSize = 0; //set by initCUDA
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/**
 * Initializes the device and sets up the block sizes and prints out memory properties
 * 
 */
void gpuCode::initCUDA(){
  int deviceCount, device;
    int gpuDeviceCount = 0;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp * properties = new cudaDeviceProp[deviceCount];
    
    if (cudaResultCode != cudaSuccess) 
        deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties[device], device);
        if (properties[device].major != 9999) /* 9999 means emulation only */
            ++gpuDeviceCount;
    }
    std::cout << gpuDeviceCount << " GPU CUDA device(s) found" << std::endl;
    if (gpuDeviceCount == 0){
	std::cout << "FATAL: NO CUDA CAPABLE CARDS ON THIS SYSTEM" << std::endl;
        exit(1);
    }
    cudaSetDevice(0);
    gpuBlockSize = 256;
    size_t free = 0;
    size_t total = 0;
    cudaMemGetInfo(&free,&total);
    std::cout << "Total GPU Memory on card: " << total/1024/1024 << " MB" << std::endl;
    std::cout << "Total GPU Memory available: " << free/1024/1024 << " MB" << std::endl;
    delete[] properties;
}
/**
 * Releases and resets GPU
 * 
 */
void gpuCode::releaseCard(){
  CUDA_CHECK_RETURN(cudaDeviceReset());
}
/*
 * Inits the compressor
 * @params iv the first dataframe that serves as a basis for the compresson of further dataframes
 * @params ivlength the length of the iv vector
 * @throws invalidInitializationException if the IV is empty
 */
void gpuCode::compressor::initCompressor(const float* iv, uint64_t ivLength){
  if (ivLength < 1)
    throw invalidInitializationException();
  uint32_t numBlocks = (ivLength / gpuBlockSize) + (ivLength%gpuBlockSize != 0); //+1 iff there is remaining elements after number of completely fulled blocks
  if (_gpuCompressorIV != NULL)
    CUDA_CHECK_RETURN(cudaFree(_gpuCompressorIV));
  CUDA_CHECK_RETURN(cudaMalloc((void**) &_gpuCompressorIV, sizeof(uint32_t) * numBlocks * gpuBlockSize));
  CUDA_CHECK_RETURN(cudaMemset(_gpuCompressorIV,0,sizeof(uint32_t) * numBlocks * gpuBlockSize)); //ensure padding is set to zero on the device
  CUDA_CHECK_RETURN(cudaMemcpy(_gpuCompressorIV, iv, ivLength*sizeof(float), cudaMemcpyHostToDevice));
  _gpuCompressorIVLength = ivLength;
  _gpuCompressorAccumulatedTime = 0;
  _gpuAccumCompressedDataSize = ivLength+1;
}

/*
 * Releases resources held by the compressor
 */
void gpuCode::compressor::releaseResources(){
  if (_gpuCompressorIV != NULL){
    CUDA_CHECK_RETURN(cudaFree((void*) _gpuCompressorIV));
    _gpuCompressorIV = NULL;
    _gpuCompressorIVLength = -1;
  }
}

/*
 * Gets the accumulated time since compressor initialization
 */
double gpuCode::compressor::getAccumulatedRunTimeSinceInit(){
  return _gpuCompressorAccumulatedTime;
}

/*
 * Gets the accumulated size needed to store the data since initialization of the compressor 
 * (this can be used to compute the compression ratio)
 */
uint32_t gpuCode::compressor::getAccumulatedCompressedDataSize(){
  return _gpuAccumCompressedDataSize;
}

const uint32_t BYTESPERINTMIN1_2 = sizeof(uint32_t) * sizeof(uint8_t) - 1;
inline int32_t imax_2( int32_t x, int32_t y )
{
    return x - ((x - y) & ((x - y) >> (BYTESPERINTMIN1_2)));
}
inline int32_t imin_2( int32_t x, int32_t y )
{
    return y ^ ((x ^ y) & -(x < y)); // min(x, y)
}

__device__ void storePrefixStream(const uint32_t * iv, uint32_t elementCount, uint32_t chunkSize, 
			uint32_t * residualAndPrefixStore, 
			uint32_t lowerBound,uint32_t blockThreadId,uint32_t index,uint32_t element,
			uint32_t bankOffset, uint32_t prefixArrOffset, uint32_t numBlocks,uint32_t prefix){
    extern __shared__ uint32_t counts[]; //the kernel must be called with "length" as a third special arguement  
    
    uint32_t lshiftAmountPrefixes = gpuStorageIndiceCapacity - gpuBitCountForRepresentation;
    if ((index < elementCount)){
        //save the prefixes:
        uint32_t iTimesgpuBitCountForRepresentation = blockThreadId*gpuBitCountForRepresentation;
        uint32_t startingIndex = (iTimesgpuBitCountForRepresentation) >> 5;
        uint32_t rshiftAmount = (iTimesgpuBitCountForRepresentation) % gpuStorageIndiceCapacity;
	atomicOr(residualAndPrefixStore + numBlocks + prefixArrOffset + blockIdx.x * ((chunkSize * gpuBitCountForRepresentation) / gpuStorageIndiceCapacity + 1) + startingIndex,
		 ((prefix << lshiftAmountPrefixes) >> rshiftAmount)); //according to the cuda developer guide this will compute the or and store it back to the same address
    }
}

/**
 * Define a macro according to Nvidia's GPU Gems 3 to offset the indexing in order to avoid bank conflicts
 * Compute Capability 	1.x: 16 banks
 * 			2.x/3.x: 32 banks
 */
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \ ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__device__ void computeScan(uint32_t n, uint32_t blockSize) {
	extern __shared__ uint32_t counts[]; //the kernel must be called with "length" as a third special arguement
	int thid = threadIdx.x; //TODO::use a more complex indexing scheme to deal with larger arrays
	int blockOffset = blockIdx.x*blockSize;
	int offset = 1;
	//up-sweep:
	for (int d = blockSize>>1; d > 0; d >>= 1)                    // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d){
			int thidTimes2 = (thid<<1);
		    int ai = offset*(thidTimes2+1)-1;
		    int bi = offset*(thidTimes2+2)-1;
		    ai += CONFLICT_FREE_OFFSET(ai);
		    bi += CONFLICT_FREE_OFFSET(bi);
		    counts[bi] += counts[ai];
		}
		offset <<= 1;
	}
	//clear:
	if (thid==0) {
		int iMax = blockSize - 1;
		counts[iMax + CONFLICT_FREE_OFFSET(iMax)] = 0;
	}
	//down-sweep:
	for (int d = 1; d < blockSize; d <<= 1) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d){
			int thidTimes2 = (thid<<1);
			int ai = offset*(thidTimes2+1)-1;
			int bi = offset*(thidTimes2+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			uint32_t t = counts[ai];
			counts[ai] = counts[bi];
			counts[bi] += t;
		}
	}
}

__device__ uint32_t storeResidualStream(uint32_t elementCount, uint32_t chunkSize, 
			uint32_t * residualAndPrefixStore, 
			uint32_t lowerBound,uint32_t blockThreadId,uint32_t index,uint32_t xoredElement,
			uint32_t bankOffset, uint32_t numBlocks){
  extern __shared__ uint32_t counts[]; //the kernel must be called with "length" as a third special arguement  
  if ((index < elementCount)){ //exclude the last element
	//save the residuals:
        uint32_t countIndexN = blockThreadId+bankOffset;
        uint32_t accumulatedIndex = counts[countIndexN];
	uint32_t count = counts[(chunkSize<<1) + countIndexN]; //get the original count value before the prefix scan was computed
        uint32_t startingIndex = accumulatedIndex >> 5;
        uint8_t lshiftAmount = (gpuStorageIndiceCapacity - count);
        uint32_t rshiftAmount = accumulatedIndex % gpuStorageIndiceCapacity;
        uint8_t writtenBits = gpuStorageIndiceCapacity - lshiftAmount - max(rshiftAmount-lshiftAmount,0);
        atomicOr(residualAndPrefixStore + numBlocks +  blockIdx.x * (chunkSize + 1) + startingIndex,((xoredElement << lshiftAmount) >> rshiftAmount));
        atomicOr(residualAndPrefixStore + numBlocks + blockIdx.x * (chunkSize + 1) + startingIndex + 1,(xoredElement << (lshiftAmount + writtenBits - 1) << 1));    
    }
}
__global__ void gpuCompressionKernel(const uint32_t * data, uint32_t * iv, uint32_t elementCount, uint32_t chunkSize, 
			uint32_t * residualAndPrefixStore, uint32_t prefixArrOffset, uint32_t numBlocks){
      extern __shared__ uint32_t counts[]; //the kernel must be called with "length" as a third special arguement
    
    //Create difference array, count used bits (up to 3 bytes of leading zeros) and save prefixes
    uint32_t lowerBound = blockIdx.x*chunkSize;
    uint32_t blockThreadId = threadIdx.x;
    //in line with the way GPU Gems 3 structures the parallel prefix sum we have to copy TWO data elements into registers
    uint32_t ai = blockThreadId;
    uint32_t bi = blockThreadId + (chunkSize>>1);
    uint32_t bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    uint32_t bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    uint32_t  blockOffsetA = lowerBound + ai;
    uint32_t  blockOffsetB = lowerBound + bi;
    uint32_t dataElementA = NULL;
    uint32_t dataElementB = NULL;
    uint32_t lastElemVal = 0;
    uint32_t xoredElementA = 0;
    uint32_t xoredElementB = 0;
    uint32_t prefixA = 0;
    uint32_t prefixB = 0;
    if (blockOffsetA < elementCount){
      uint32_t ivElement = iv[blockOffsetA];
      dataElementA = data[blockOffsetA];
      xoredElementA = ivElement ^ dataElementA;
      prefixA = min(3,__clz(xoredElementA) >> 3);
      //store a copy of the orignal count at an 2* BLOCK SIZE offset in shared memory so that we can get the scan values and originals later!
      uint32_t countIndexN = ai+bankOffsetA;
      counts[countIndexN] = ((sizeof(uint32_t)-prefixA) << 3);
      counts[(chunkSize<<1)+countIndexN] = counts[countIndexN];
    }
    if (blockOffsetB < elementCount){
      uint32_t ivElement = iv[blockOffsetB];
      dataElementB = data[blockOffsetB];
      xoredElementB = ivElement ^ dataElementB;
      prefixB = min(3,__clz(xoredElementB) >> 3);
      //store a copy of the orignal count at an 2* BLOCK SIZE offset in shared memory so that we can get the scan values and originals later!
      uint32_t countIndexN = bi+bankOffsetB;
      counts[countIndexN] = ((sizeof(uint32_t)-prefixB) << 3);
      counts[(chunkSize<<1)+countIndexN] = counts[countIndexN];
    }
    __syncthreads();
    //compute lzc and save the prefixes:
     storePrefixStream(iv,elementCount,chunkSize,residualAndPrefixStore,lowerBound,ai,blockOffsetA,xoredElementA,bankOffsetA,prefixArrOffset,numBlocks,prefixA);
     storePrefixStream(iv,elementCount,chunkSize,residualAndPrefixStore,lowerBound,bi,blockOffsetB,xoredElementB,bankOffsetB,prefixArrOffset,numBlocks,prefixB);
    __syncthreads();
    //compute parallel prefix sum (this method taken from GPU GEMS 3 computes 2 elements at a time):
     computeScan(elementCount,chunkSize);
     __syncthreads();
    //now save the residuals:
    storeResidualStream(elementCount,chunkSize,residualAndPrefixStore,lowerBound,ai,blockOffsetA,xoredElementA,bankOffsetA,numBlocks);
    storeResidualStream(elementCount,chunkSize,residualAndPrefixStore,lowerBound,bi,blockOffsetB,xoredElementB,bankOffsetB,numBlocks);
    __syncthreads();
    //calculate storage space used by residuals:
    if (blockOffsetA == elementCount-1){ //last element before end of block
       uint32_t accumulatedIndex = counts[bankOffsetA+ai] + counts[(chunkSize << 1) + bankOffsetA+ai];
       uint32_t sizeOfResidualArray = accumulatedIndex / gpuStorageIndiceCapacity +
                           (accumulatedIndex % gpuStorageIndiceCapacity != 0);
        residualAndPrefixStore[blockIdx.x] = sizeOfResidualArray; 
    } else if (blockOffsetB == elementCount-1){
       uint32_t accumulatedIndex = counts[bankOffsetB+bi] + counts[(chunkSize << 1) + bankOffsetB+bi];
       uint32_t sizeOfResidualArray = accumulatedIndex / gpuStorageIndiceCapacity +
                           (accumulatedIndex % gpuStorageIndiceCapacity != 0);
        residualAndPrefixStore[blockIdx.x] = sizeOfResidualArray; 
    } else if (blockThreadId == (chunkSize >> 1)-1){ //last thread of block
       uint32_t lastCountElemIndex = bankOffsetB+bi;
      uint32_t accumulatedIndex = counts[lastCountElemIndex] + counts[(chunkSize << 1) + lastCountElemIndex];
      uint32_t sizeOfResidualArray = accumulatedIndex / gpuStorageIndiceCapacity +
                          (accumulatedIndex % gpuStorageIndiceCapacity != 0);
	residualAndPrefixStore[blockIdx.x] = sizeOfResidualArray; 
    }
    //Copy the current data to the IV memory for the next round of compression
    if (blockOffsetA < elementCount)
      iv[blockOffsetA] = data[blockOffsetA];
    if (blockOffsetB < elementCount)
      iv[blockOffsetB] = data[blockOffsetB];
}

void gpuDecompressionKernel(uint32_t chunkSize, uint32_t dataBlockSize, 
			  uint32_t * compressedPrefixes, uint32_t * compressedResiduals,
			  uint32_t dataBlockIndex,uint32_t lowerBound) {
    uint32_t accumulatedIndex = 0;
    uint8_t lshiftAmount = (gpuStorageIndiceCapacity - gpuBitCountForRepresentation);
    for (uint32_t i = 0; i < dataBlockSize; ++i) {
	//inflate prefix
	uint32_t prefixIndex = i*gpuBitCountForRepresentation;
        uint32_t startingIndex = prefixIndex >> 5;
        uint8_t rshiftAmount = prefixIndex % gpuStorageIndiceCapacity;
        uint8_t prefix = ((compressedPrefixes[startingIndex] << rshiftAmount) >> lshiftAmount);
        uint32_t count = gpuStorageIndiceCapacity - (prefix << 3);
	//inflate its associated residual
	startingIndex = accumulatedIndex >> 5;
        uint8_t residuallshiftAmount = (gpuStorageIndiceCapacity - count);
        rshiftAmount = accumulatedIndex % gpuStorageIndiceCapacity;
        uint8_t writtenBits = gpuStorageIndiceCapacity - residuallshiftAmount - imax_2(rshiftAmount-residuallshiftAmount,0);
        register uint32_t residual = ( (compressedResiduals[startingIndex] << rshiftAmount) >> residuallshiftAmount);
        residual |= 
	  ( compressedResiduals[startingIndex+(gpuStorageIndiceCapacity - residuallshiftAmount - writtenBits > 0)] >> (residuallshiftAmount + writtenBits - 1) >> 1);
        _gpuDecompressorIV[lowerBound+i] ^= residual;
        accumulatedIndex += count;
    }
}


/*
 * Compresses a dataframe. This function will compress a dataframe in parallel and will call back with the compressed data when completed.
 * The user should save the initialization vector dataframe and the elementCount to file himself. For dataframe index > 1 the user
 * should save the compressed prefix and residual array to persistent storage within the scope of the callback function. After the
 * return of the callback function the compressed data will be deleted from memory and the pointers will no longer be valid.
 * @throws invalidInitializationException if the length of the dataframe vector does not match the length of the initialization vector
 */

void gpuCode::compressor::compressData(const float * data, uint32_t elementCount,
			  void (*callBack)(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, uint32_t chunkCount, uint32_t * chunkSizes)){
    if (_gpuCompressorIV == NULL || _gpuCompressorIVLength != elementCount)
        throw invalidInitializationException();
    uint32_t chunkSize = gpuBlockSize; 
    uint32_t numStores = elementCount/gpuBlockSize + (elementCount%gpuBlockSize != 0); //+1 iff there is remaining elements after number of completely fulled blocks
    uint32_t** residlualStore = new uint32_t*[numStores];
    uint32_t** prefixStore = new uint32_t*[numStores];
    uint32_t* residualSizesStore = new uint32_t[numStores];
    uint32_t* prefixSizesStore = new uint32_t[numStores];
    uint32_t* chunkSizes = new uint32_t[numStores];
    //alloc space for the data on the card
    uint32_t* gpuData = NULL;
    CUDA_CHECK_RETURN(cudaMalloc((void**) &gpuData, sizeof(uint32_t) * numStores * gpuBlockSize)); //pad so that we completely fill every SM
    CUDA_CHECK_RETURN(cudaMemcpy(gpuData, data, elementCount*sizeof(float), cudaMemcpyHostToDevice));
    timer::tic();
    //alloc space for residuals and prefixes on the card
    uint32_t* gpuResidualsAndPrefixesStore = NULL;
    uint32_t* gpuPrefixSizesStore = NULL;    
    //create the wrappers for the residual and prefix memory stores which will be sent to the kernel: 
    uint32_t sizeOfPrefixArray = sizeof(uint32_t)*((elementCount * gpuBitCountForRepresentation) / gpuStorageIndiceCapacity + elementCount); //pad the prefix array with 1 for every block to prevent branch diversion in the kernel
    uint32_t sizeOfResidualArray = sizeof(uint32_t)*(chunkSize*numStores + elementCount); //pad the residual array with 1 to prevent branch diversion in the kernel
    uint32_t sizeOfResidualSizesArray = sizeof(uint32_t) * numStores;
    uint32_t prefixArrOffset = (chunkSize*numStores + elementCount);
    CUDA_CHECK_RETURN(cudaMalloc((void**) &gpuResidualsAndPrefixesStore, sizeOfResidualArray+sizeOfPrefixArray+sizeOfResidualSizesArray));
    CUDA_CHECK_RETURN(cudaMemset(gpuResidualsAndPrefixesStore,0,sizeOfPrefixArray+sizeOfResidualArray+sizeOfResidualSizesArray));
   
    gpuCompressionKernel<<<numStores, (chunkSize)/2, (chunkSize) * 4 * sizeof(uint32_t)>>>(gpuData,
			_gpuCompressorIV,elementCount,chunkSize,
			gpuResidualsAndPrefixesStore,prefixArrOffset,numStores);
    cudaThreadSynchronize();
    CUDA_CHECK_RETURN(cudaGetLastError());
    //copy the padded prefix and residual arrays over so that it can be bundled into 2 transactions and not hundreds:
//     uint32_t * tempResidualsAndPrefixesStore = NULL;
//     CUDA_CHECK_RETURN(cudaMallocHost((void**)&tempResidualsAndPrefixesStore,sizeOfResidualArray+sizeOfPrefixArray+sizeOfResidualSizesArray,0));
    uint32_t * tempResidualsAndPrefixesStore = (uint32_t*) malloc(sizeOfResidualArray+sizeOfPrefixArray+sizeOfResidualSizesArray);
    CUDA_CHECK_RETURN(cudaMemcpy(tempResidualsAndPrefixesStore,gpuResidualsAndPrefixesStore, 
				   sizeOfResidualArray+sizeOfPrefixArray+sizeOfResidualSizesArray, cudaMemcpyDeviceToHost));
    _gpuCompressorAccumulatedTime += timer::toc();
    //Now split the cuda memory up into unpadded chunks:
    uint32_t offsetPrefixes = 0;
    uint32_t offsetResiduals = 0;
    memcpy(residualSizesStore,tempResidualsAndPrefixesStore,sizeOfResidualSizesArray);
    for (uint32_t i = 0; i < numStores; ++i){      
      uint32_t elementsInDataBlock = (((i + 1)*chunkSize <= elementCount) ? chunkSize : chunkSize-((i + 1)*chunkSize-elementCount));
      uint32_t sizeOfPrefixArray = (elementsInDataBlock * gpuBitCountForRepresentation) / gpuStorageIndiceCapacity +
                                 ((elementsInDataBlock * gpuBitCountForRepresentation) % gpuStorageIndiceCapacity != 0);		 
      //  std::cout << "Copying store " << i+1 << " of " << numStores << ", with prefix array length "<< sizeOfPrefixArray <<" and residual array length " << residualSizesStore[i] << std::endl;
      prefixStore[i] = new uint32_t[sizeOfPrefixArray];
      residlualStore[i] = new uint32_t[residualSizesStore[i]];
      prefixSizesStore[i] = sizeOfPrefixArray;
      chunkSizes[i] = elementsInDataBlock;
      memcpy(prefixStore[i],tempResidualsAndPrefixesStore + numStores + prefixArrOffset + offsetPrefixes, sizeof(uint32_t) * sizeOfPrefixArray);
      memcpy(residlualStore[i],tempResidualsAndPrefixesStore + numStores + offsetResiduals, sizeof(uint32_t) * residualSizesStore[i]);
      offsetPrefixes += ((chunkSize * gpuBitCountForRepresentation) / gpuStorageIndiceCapacity + 1);
      offsetResiduals += (chunkSize + 1);
      _gpuAccumCompressedDataSize += tempResidualsAndPrefixesStore[i] + prefixSizesStore[i] + 1;
    }
    //_gpuCompressorAccumulatedTime += timer::toc();
//     cudaFreeHost(tempResidualsAndPrefixesStore);
    free(tempResidualsAndPrefixesStore);
    callBack(elementCount,residualSizesStore,residlualStore,prefixSizesStore,prefixStore,numStores,chunkSizes);
    for (uint32_t i = 0; i < numStores; ++i){
       delete[] residlualStore[i];
       delete[] prefixStore[i];
    }
    CUDA_CHECK_RETURN(cudaFree((void*) gpuData));
    CUDA_CHECK_RETURN(cudaFree((void*) gpuResidualsAndPrefixesStore));
    delete[] residlualStore;
    delete[] prefixStore;
    delete[] residualSizesStore;
    delete[] prefixSizesStore;
    delete[] chunkSizes;
}

/*
 * Inits the decompressor
 * @params iv the first dataframe that serves as a basis for the decompresson of further dataframes
 * @params ivlength the length of the iv vector
 * @throws invalidInitializationException if the IV is empty
 */
void gpuCode::decompressor::initDecompressor(const float* iv, uint64_t ivLength){
  if (ivLength < 1)
    throw invalidInitializationException();
  if (_gpuDecompressorIV != NULL)
    delete[] _gpuDecompressorIV;
  _gpuDecompressorIV = new uint32_t[ivLength];
  memcpy(_gpuDecompressorIV,iv,ivLength*sizeof(float));
  _gpuDecompressorIVLength = ivLength;
  _gpuDecompressorAccumulatedTime = 0;
  _gpuAccumDecompressedDataSize = ivLength;
}

/*
 * Releases resources held by the decompressor
 */
void gpuCode::decompressor::releaseResources(){
  if (_gpuDecompressorIV != NULL){
    delete[] _gpuDecompressorIV;
    _gpuDecompressorIV = NULL;
    _gpuDecompressorIVLength = -1;
  }
}

/*
 * Gets the accumulated time since decompressor initialization
 */
double gpuCode::decompressor::getAccumulatedRunTimeSinceInit(){
  return _gpuDecompressorAccumulatedTime;
}

/*
 * Gets the accumulated size of the decompressed data since initialization 
 * (this can be used to compute the compression ratio)
 */
uint32_t gpuCode::decompressor::getAccumulatedDecompressedDataSize(){
  return _gpuAccumDecompressedDataSize;
}

/*
 * Decompresses a dataframe. This function will decompress a dataframe in parallel and will call back with the decompressed data when completed.
 * The user should save the initialization vector dataframe and the elementCount to file himself. For dataframe index > 1 the user
 * should save the decompressed frame to persistent storage within the scope of the callback function. After the
 * return of the callback function the decompressed data will be deleted from memory and the pointers will no longer be valid.
 * @throws invalidInitializationException if the length of the dataframe vector does not match the length of the initialization vector
 */
void gpuCode::decompressor::decompressData(uint32_t elementCount, uint32_t chunkCount, uint32_t * chunkSizes, 
			uint32_t ** compressedResiduals, uint32_t ** compressedPrefixes, 
			void (*callBack)(uint32_t elementCount, uint32_t * decompressedData)){
  if (_gpuDecompressorIV == NULL || _gpuDecompressorIVLength != elementCount)
        throw invalidInitializationException();
  timer::tic();
  #pragma omp parallel for 
  for (uint32_t dataBlockIndex = 0; dataBlockIndex < chunkCount; ++dataBlockIndex) {
    gpuDecompressionKernel(chunkSizes[0],chunkSizes[dataBlockIndex],compressedPrefixes[dataBlockIndex],
			compressedResiduals[dataBlockIndex],dataBlockIndex,dataBlockIndex*chunkSizes[0]);
  }
  _gpuDecompressorAccumulatedTime += timer::toc();
  callBack(elementCount, _gpuDecompressorIV);
  _gpuAccumDecompressedDataSize += elementCount;
}
