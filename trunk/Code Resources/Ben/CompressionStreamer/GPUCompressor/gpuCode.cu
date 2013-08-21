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

void initCUDA(){
  int deviceCount, device;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess) 
        deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) /* 9999 means emulation only */
            ++gpuDeviceCount;
    }
    std::cout << gpuDeviceCount << " GPU CUDA device(s) found" << std::endl;

    /* don't just return the number of gpus, because other runtime cuda
       errors can also yield non-zero return values */
    if (gpuDeviceCount == 0){
	std::cout << "FATAL: NO CUDA CAPABLE CARDS ON THIS SYSTEM" << std::endl;
        exit(1);
    }
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
  if (_gpuCompressorIV != NULL)
    _gpuCompressorIV = new uint32_t[ivLength];
  _gpuCompressorIV = new uint32_t[ivLength];
  memcpy(_gpuCompressorIV,iv,ivLength*sizeof(float));
  _gpuCompressorIVLength = ivLength;
  _gpuCompressorAccumulatedTime = 0;
  _gpuAccumCompressedDataSize = ivLength+1;
}

/*
 * Releases resources held by the compressor
 */
void gpuCode::compressor::releaseResources(){
  if (_gpuCompressorIV != NULL){
    free(_gpuCompressorIV);
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

void gpuCompressionKernel(const float * data, uint32_t elementCount, uint32_t dataBlockIndex, 
			uint32_t chunkSize, uint32_t ** prefixStore, uint32_t ** residualStore, 
			uint32_t * prefixSizeStore, uint32_t * residualSizeStore, uint32_t * dataBlockSizes,
			uint32_t lowerBound, uint32_t elementsInDataBlock){

//     //create storage for counts and prefixes:
//     uint32_t sizeOfPrefixArray = (elementsInDataBlock * gpuBitCountForRepresentation) / gpuStorageIndiceCapacity +
//                                  ((elementsInDataBlock * gpuBitCountForRepresentation) % gpuStorageIndiceCapacity != 0);
//     uint32_t * arrPrefix = new uint32_t[sizeOfPrefixArray];
//     memset(arrPrefix,0,sizeof(uint32_t)*sizeOfPrefixArray);
//     uint32_t * arrResiduals = new uint32_t[elementsInDataBlock+1]; //this padding actually waste less space than having a count array, +1 to avoid a branch later on when writing the remainder of the residuals
//     memset(arrResiduals,0,sizeof(uint32_t)*elementsInDataBlock);
// 
//     //Create difference array, count used bits (up to 3 bytes of leading zeros) and save prefixes
//     uint32_t lshiftAmountPrefixes = gpuStorageIndiceCapacity - gpuBitCountForRepresentation;
//     uint32_t accumulatedIndex = 0;
//     for (uint32_t i = 0; i < elementsInDataBlock; ++i) {
// 	uint32_t index = i+lowerBound;
//         //save the prefixes:
//         uint32_t element = (_gpuCompressorIV[index] ^= ((uint32_t*)&(data[0]))[index]);
// 	uint32_t prefix0 = imin(3,lzc (element) >> 3);
//         uint32_t iTimesgpuBitCountForRepresentation = i*gpuBitCountForRepresentation;
//         uint32_t startingIndex = (iTimesgpuBitCountForRepresentation) >> 5;
//         uint32_t rshiftAmount = (iTimesgpuBitCountForRepresentation) % gpuStorageIndiceCapacity;
//         arrPrefix[startingIndex] |= ((prefix0 << lshiftAmountPrefixes) >> rshiftAmount);
//         uint32_t count = ((sizeof(uint32_t)-prefix0) << 3);
//         
// 	//save the residuals:
//         startingIndex = accumulatedIndex >> 5;
//         uint8_t lshiftAmount = (gpuStorageIndiceCapacity - count);
//         rshiftAmount = accumulatedIndex % gpuStorageIndiceCapacity;
//         uint8_t writtenBits = gpuStorageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
//         element = _gpuCompressorIV[index]; //it seems after _lzcnt_u32 touches a memory location it is not optimized correctly this is a work arround
//         arrResiduals[startingIndex] |= ( (element << lshiftAmount) >> rshiftAmount);
//         arrResiduals[startingIndex+1] |= (element << (lshiftAmount + writtenBits - 1) << 1);
//         accumulatedIndex += count;
//     }
//  
//     //calculate storage space used by residuals:
//     uint32_t sizeOfResidualArray = accumulatedIndex / gpuStorageIndiceCapacity +
//                           (accumulatedIndex % gpuStorageIndiceCapacity != 0);
// 
//     //Store pointers to the current prefixes and residuals
//     residualStore[dataBlockIndex] = arrResiduals;
//     prefixStore[dataBlockIndex] = arrPrefix;
//     residualSizeStore[dataBlockIndex] = sizeOfResidualArray;
//     prefixSizeStore[dataBlockIndex] = sizeOfPrefixArray; 
//     dataBlockSizes[dataBlockIndex] = elementsInDataBlock;
//     
//     
//     //Copy the current data to the IV memory for the next round of compression
//      memcpy(_gpuCompressorIV+lowerBound,data+lowerBound,elementsInDataBlock*sizeof(float));
//     //the prefixes and residluals will be freed by the caller
}


void gpuDecompressionKernel(uint32_t chunkSize, uint32_t dataBlockSize, 
			  uint32_t * compressedPrefixes, uint32_t * compressedResiduals,
			  uint32_t dataBlockIndex,uint32_t lowerBound) {
//     uint32_t accumulatedIndex = 0;
//     uint8_t lshiftAmount = (gpuStorageIndiceCapacity - gpuBitCountForRepresentation);
//     for (uint32_t i = 0; i < dataBlockSize; ++i) {
// 	//inflate prefix
// 	uint32_t prefixIndex = i*gpuBitCountForRepresentation;
//         uint32_t startingIndex = prefixIndex >> 5;
//         uint8_t rshiftAmount = prefixIndex % gpuStorageIndiceCapacity;
//         uint8_t prefix = ((compressedPrefixes[startingIndex] << rshiftAmount) >> lshiftAmount);
//         uint32_t count = gpuStorageIndiceCapacity - (prefix << 3);
// 	//inflate its associated residual
// 	startingIndex = accumulatedIndex >> 5;
//         uint8_t residuallshiftAmount = (gpuStorageIndiceCapacity - count);
//         rshiftAmount = accumulatedIndex % gpuStorageIndiceCapacity;
//         uint8_t writtenBits = gpuStorageIndiceCapacity - residuallshiftAmount - imax(rshiftAmount-residuallshiftAmount,0);
//         register uint32_t residual = ( (compressedResiduals[startingIndex] << rshiftAmount) >> residuallshiftAmount);
//         residual |= 
// 	  ( compressedResiduals[startingIndex+(gpuStorageIndiceCapacity - residuallshiftAmount - writtenBits > 0)] >> (residuallshiftAmount + writtenBits - 1) >> 1);
//         _gpuDecompressorIV[lowerBound+i] ^= residual;
//         accumulatedIndex += count;
//     }
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
    timer::tic();
    uint32_t NUMTHREADS = omp_get_max_threads();
    uint32_t chunkSize = elementCount/NUMTHREADS; 
    uint32_t numStores = NUMTHREADS+(elementCount%NUMTHREADS != 0);
    uint32_t** residlualStore = new uint32_t*[numStores];
    uint32_t** prefixStore = new uint32_t*[numStores];
    uint32_t* residualSizesStore = new uint32_t[numStores];
    uint32_t* prefixSizesStore = new uint32_t[numStores];
    uint32_t* chunkSizes = new uint32_t[numStores];
#pragma omp parallel for 
    for (uint32_t dataBlockIndex = 0; dataBlockIndex < NUMTHREADS; ++dataBlockIndex) {  
      gpuCompressionKernel(data,elementCount,dataBlockIndex,chunkSize,prefixStore,
			residlualStore,prefixSizesStore,residualSizesStore,chunkSizes,
			dataBlockIndex*chunkSize,
			(((dataBlockIndex + 1)*chunkSize <= elementCount) ? chunkSize : chunkSize-((dataBlockIndex + 1)*chunkSize-elementCount)));
    }
      gpuCompressionKernel(data,elementCount,NUMTHREADS,chunkSize,prefixStore,
			residlualStore,prefixSizesStore,residualSizesStore,chunkSizes,
			NUMTHREADS*chunkSize,
			(((NUMTHREADS + 1)*chunkSize <= elementCount) ? chunkSize : chunkSize-((NUMTHREADS + 1)*chunkSize-elementCount)));
    _gpuCompressorAccumulatedTime += timer::toc();
    //Now do the callback and free all resources afterwards except the IV:
    for (uint32_t i = 0; i < numStores; ++i){
      _gpuAccumCompressedDataSize += residualSizesStore[i] + prefixSizesStore[i] + 1;
    }
    callBack(elementCount,residualSizesStore,residlualStore,prefixSizesStore,prefixStore,numStores,chunkSizes);
    for (uint32_t i = 0; i < numStores; ++i){
       free(residlualStore[i]);
       free(prefixStore[i]);
    }
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
