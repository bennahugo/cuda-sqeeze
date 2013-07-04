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


#include "cpuCode.h"
uint32_t * _compressorIV = NULL; 
uint64_t _compressorIVLength = -1;
uint32_t _accumCompressedDataSize = 0;
uint32_t * _decompressorIV = NULL; 
uint64_t _decompressorIVLength = -1;
double _compressorAccumulatedTime = 0;
double _decompressorAccumulatedTime = 0;
uint32_t _accumDecompressedDataSize = 0;
const uint8_t storageIndiceCapacity = 8*sizeof(uint32_t);
const uint8_t bitCountForRepresentation = 2;

// Define macro for aligning data
#ifdef _MSC_VER
// If Microsoft compiler
#define Alignd(X) __declspec(align(16)) X
#else
// Gnu compiler, etc.
#define Alignd(X) X __attribute__((aligned(16)))
#endif
inline int32_t imax( int32_t a, int32_t b )
{
    return a + ( ( b - a ) & ( (a - b) >> 31 ) );
}
/*
 * Inits the compressor
 * @params iv the first dataframe that serves as a basis for the compresson of further dataframes
 * @params ivlength the length of the iv vector
 * @throws invalidInitializationException if the IV is empty
 */
void cpuCode::compressor::initCompressor(const float* iv, uint64_t ivLength){
  if (ivLength < 1)
    throw invalidInitializationException();
  if (_compressorIV != NULL)
    _compressorIV = (uint32_t*)_mm_malloc(sizeof(uint32_t)*ivLength,16);
  _compressorIV = new uint32_t[ivLength];
  memcpy(_compressorIV,iv,ivLength*sizeof(float));
  _compressorIVLength = ivLength;
  _compressorAccumulatedTime = 0;
  _accumCompressedDataSize = ivLength+1;
}

/*
 * Releases resources held by the compressor
 */
void cpuCode::compressor::releaseResources(){
  if (_compressorIV != NULL){
    _mm_free(_compressorIV);
    _compressorIV = NULL;
    _compressorIVLength = -1;
  }
}

/*
 * Gets the accumulated time since compressor initialization
 */
double cpuCode::compressor::getAccumulatedRunTimeSinceInit(){
  return _compressorAccumulatedTime;
}

/*
 * Gets the accumulated size needed to store the data since initialization of the compressor 
 * (this can be used to compute the compression ratio)
 */
uint32_t cpuCode::compressor::getAccumulatedCompressedDataSize(){
  return _accumCompressedDataSize;
}

struct compressionKernelArgs {
  const float * data; 
  uint32_t elementCount;
  uint32_t dataBlockIndex; 
  uint32_t chunkSize;
  uint32_t ** prefixStore;
  uint32_t ** residualStore;
  uint32_t * prefixSizeStore;
  uint32_t * residualSizeStore;
  uint32_t * dataBlockSizes;
};

void printBinaryRepresentationT(void * data, int sizeInBytes){
  using namespace std;
  char * temp = (char *)data;
  for (int i = sizeInBytes - 1; i >= 0; --i){
    for (int b = 7; b >= 0; --b)  
      cout << (0x1 << b & temp[i] ? '1' : '0');
    cout << ' ';
  }
  cout << endl;
}

void* compressionKernel(void * args){
    const float * data = ((compressionKernelArgs*)args)->data;
    uint32_t elementCount = ((compressionKernelArgs*)args)->elementCount;
    uint32_t dataBlockIndex = ((compressionKernelArgs*)args)->dataBlockIndex; 
    uint32_t chunkSize = ((compressionKernelArgs*)args)->chunkSize;
    
    uint32_t lowerBound = dataBlockIndex*chunkSize;
    uint32_t elementsInDataBlock = (((dataBlockIndex + 1)*chunkSize <= elementCount) ? chunkSize : chunkSize-((dataBlockIndex + 1)*chunkSize-elementCount));
    /*
     * create storage for counts and prefixes:
     */
    uint32_t sizeOfPrefixArray = (elementsInDataBlock * bitCountForRepresentation) / storageIndiceCapacity +
                                 ((elementsInDataBlock * bitCountForRepresentation) % storageIndiceCapacity != 0 ? 1 : 0);
    uint32_t * arrPrefix = (uint32_t*)_mm_malloc(sizeof(uint32_t)*sizeOfPrefixArray,16); //default initialize
    for (int i = 0; i < sizeOfPrefixArray; ++i) {
	arrPrefix[i] = 0;
    }
    uint32_t * arrCounts = (uint32_t*)_mm_malloc(sizeof(uint32_t)*elementsInDataBlock,16); //no need to initialize we're going to override this in any case
    uint32_t sizeOfResidualArray = 0;

    /*
     * Create difference array, count used bits (up to 3 bytes of leading zeros) and save prefixes
     */
    uint32_t lshiftAmount = storageIndiceCapacity - bitCountForRepresentation;
    for (uint32_t i = 0; i < elementsInDataBlock; ++i) {
 	_compressorIV[i+lowerBound] ^= ((uint32_t*)&(data[0]))[i+lowerBound]; 
        uint8_t* bytes = (uint8_t*)&(_compressorIV[i+lowerBound]);
        register uint32_t prefix0 = (bytes[3] == 0) + ((bytes[3] | bytes[2]) == 0) + ((bytes[3] | bytes[2] | bytes[1]) == 0); //count how many bytes of leading zeros
        uint32_t startingIndex = (i*bitCountForRepresentation) / storageIndiceCapacity;
        uint32_t rshiftAmount = (i*bitCountForRepresentation) % storageIndiceCapacity;
        arrPrefix[startingIndex] |= ((prefix0 << lshiftAmount) >> rshiftAmount);
        sizeOfResidualArray += (arrCounts[i] = ((4-prefix0)*8));
    }
    /*
     * create storage for residuals:
     */
    sizeOfResidualArray = sizeOfResidualArray / storageIndiceCapacity +
                          (sizeOfResidualArray % storageIndiceCapacity != 0 ? 1 : 0);
    uint32_t * arrResiduals = (uint32_t*)_mm_malloc(sizeof(uint32_t)*sizeOfResidualArray,16);
    for (int i = 0; i < sizeOfResidualArray; ++i) {
	arrResiduals[i] = 0;
    }
    /*
     * save residuals:
     */
     uint32_t accumulatedIndex = 0;
     for (uint32_t i=0; i < elementsInDataBlock; ++i) {
         uint32_t index = accumulatedIndex;
         uint32_t ivElem = _compressorIV[i+lowerBound];
         uint32_t startingIndex = accumulatedIndex / storageIndiceCapacity;
         uint8_t lshiftAmount = (storageIndiceCapacity - arrCounts[i]);
         uint8_t rshiftAmount = accumulatedIndex % storageIndiceCapacity;
         uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
         arrResiduals[startingIndex] |= ( (ivElem << lshiftAmount) >> rshiftAmount);
         if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)
             arrResiduals[startingIndex+1] |= ( ivElem << (lshiftAmount + writtenBits - 1) << 1);
         accumulatedIndex += arrCounts[i];
     }
     
    /*
     * Store pointers to the current prefixes and residuals
     */ 
    ((compressionKernelArgs*)args)->residualStore[dataBlockIndex] = arrResiduals;
    ((compressionKernelArgs*)args)->prefixStore[dataBlockIndex] = arrPrefix;
    ((compressionKernelArgs*)args)->residualSizeStore[dataBlockIndex] = sizeOfResidualArray;
    ((compressionKernelArgs*)args)->prefixSizeStore[dataBlockIndex] = sizeOfPrefixArray;
    ((compressionKernelArgs*)args)->dataBlockSizes[dataBlockIndex] = elementsInDataBlock;
    /*
     * Copy the current data to the IV memory for the next round of compression
     */
     memcpy(_compressorIV+lowerBound,data+lowerBound,elementsInDataBlock*sizeof(float));
    _mm_free(arrCounts);
    //the prefixes and residluals will be freed by the caller
    
}

struct decompressionKernelArgs {
  uint32_t chunkSize;
  uint32_t dataBlockIndex;
  uint32_t dataBlockSize;
  uint32_t * compressedPrefixes;
  uint32_t * compressedResiduals;
};

void* decompressionKernel(void * args) {
    uint32_t dataBlockIndex = ((decompressionKernelArgs*)args)->dataBlockIndex;
    uint32_t chunkSize = ((decompressionKernelArgs*)args)->chunkSize;
    uint32_t dataBlockSize = ((decompressionKernelArgs*)args)->dataBlockSize;
    uint32_t * compressedPrefixes = ((decompressionKernelArgs*)args)->compressedPrefixes;
    uint32_t * compressedResiduals = ((decompressionKernelArgs*)args)->compressedResiduals;
    uint32_t lowerBound = dataBlockIndex*chunkSize;
    /*
     * create storage for counts and decompressed data:
     */
    uint32_t * arrCounts = (uint32_t*)_mm_malloc(sizeof(uint32_t)*chunkSize,16); //no need to initialize we're going to override this in any case

    /*
     * construct count array from prefix array
     */
    uint8_t lshiftAmount = (storageIndiceCapacity - bitCountForRepresentation);
    for (uint32_t i = 0; i < dataBlockSize; ++i) {
        uint32_t startingIndex = (i*bitCountForRepresentation) / storageIndiceCapacity;
        uint8_t rshiftAmount = (i*bitCountForRepresentation) % storageIndiceCapacity;
        uint8_t prefix = ((compressedPrefixes[startingIndex] << rshiftAmount) >> lshiftAmount);
        arrCounts[i] = storageIndiceCapacity - prefix*8;
    }

    uint32_t accumulatedIndex = 0;
    for (uint32_t i = 0; i < dataBlockSize; ++i) {
        uint32_t startingIndex = accumulatedIndex / storageIndiceCapacity;
        uint8_t lshiftAmount = (storageIndiceCapacity - arrCounts[i]);
        uint8_t rshiftAmount = accumulatedIndex % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
        register uint32_t residual = ( (compressedResiduals[startingIndex] << rshiftAmount) >> lshiftAmount);
        if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)
            residual |= ( compressedResiduals[startingIndex+1] >> (lshiftAmount + writtenBits - 1) >> 1);
// 	if (i+lowerBound == 229375){
// 	
// 	std::cout << *((float*)(_compressorIV+i+lowerBound)) << std::endl;
// 	}
        _decompressorIV[lowerBound+i] ^= residual;
        accumulatedIndex += arrCounts[i];
    }

    _mm_free(arrCounts);
}


/*
 * Compresses a dataframe. This function will compress a dataframe in parallel and will call back with the compressed data when completed.
 * The user should save the initialization vector dataframe and the elementCount to file himself. For dataframe index > 1 the user
 * should save the compressed prefix and residual array to persistent storage within the scope of the callback function. After the
 * return of the callback function the compressed data will be deleted from memory and the pointers will no longer be valid.
 * @throws invalidInitializationException if the length of the dataframe vector does not match the length of the initialization vector
 */
void cpuCode::compressor::compressData(const float * data, uint32_t elementCount,
			  void (*callBack)(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, uint32_t chunkCount, uint32_t * chunkSizes)){
    if (_compressorIV == NULL || _compressorIVLength != elementCount)
        throw invalidInitializationException();
    timer::tic();
    uint32_t NUMTHREADS = omp_get_max_threads();
    uint32_t chunkSize = elementCount/NUMTHREADS; 
    uint32_t remThreads = elementCount%NUMTHREADS != 0;
    compressionKernelArgs compressionArgs[NUMTHREADS];
    uint32_t numStores = NUMTHREADS+(elementCount%NUMTHREADS != 0);
    uint32_t** residlualStore = new uint32_t*[numStores];
    uint32_t** prefixStore = new uint32_t*[numStores];
    uint32_t* residualSizesStore = new uint32_t[numStores];
    uint32_t* prefixSizesStore = new uint32_t[numStores];
    uint32_t* chunkSizes = new uint32_t[numStores];
#pragma omp parallel for 
    for (uint32_t dataBlockIndex = 0; dataBlockIndex < numStores; ++dataBlockIndex) {
      uint32_t index = dataBlockIndex % NUMTHREADS;
      compressionArgs[index].data = data;
      compressionArgs[index].elementCount = elementCount;
      compressionArgs[index].chunkSize = chunkSize;
      compressionArgs[index].dataBlockSizes = chunkSizes;
      compressionArgs[index].dataBlockIndex = dataBlockIndex;
      compressionArgs[index].prefixSizeStore = prefixSizesStore;
      compressionArgs[index].residualSizeStore = residualSizesStore;
      compressionArgs[index].prefixStore = prefixStore;
      compressionArgs[index].residualStore = residlualStore;
      compressionKernel((void *)&compressionArgs[index]);
    }
    _compressorAccumulatedTime += timer::toc();
    //Now do the callback and free all resources afterwards except the IV:
    for (int i = 0; i < numStores; ++i){
      _accumCompressedDataSize += residualSizesStore[i] + prefixSizesStore[i] + 1;
    }
    callBack(elementCount,residualSizesStore,residlualStore,prefixSizesStore,prefixStore,numStores,chunkSizes);
    for (int i = 0; i < numStores; ++i){
       _mm_free(residlualStore[i]);
       _mm_free(prefixStore[i]);
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
void cpuCode::decompressor::initDecompressor(const float* iv, uint64_t ivLength){
  if (ivLength < 1)
    throw invalidInitializationException();
  if (_decompressorIV != NULL)
    delete[] _decompressorIV;
  _decompressorIV = new uint32_t[ivLength];
  memcpy(_decompressorIV,iv,ivLength*sizeof(float));
  _decompressorIVLength = ivLength;
  _decompressorAccumulatedTime = 0;
  _accumDecompressedDataSize = ivLength;
}

/*
 * Releases resources held by the decompressor
 */
void cpuCode::decompressor::releaseResources(){
  if (_decompressorIV != NULL){
    delete[] _decompressorIV;
    _decompressorIV = NULL;
    _decompressorIVLength = -1;
  }
}

/*
 * Gets the accumulated time since decompressor initialization
 */
double cpuCode::decompressor::getAccumulatedRunTimeSinceInit(){
  return _decompressorAccumulatedTime;
}

/*
 * Gets the accumulated size of the decompressed data since initialization 
 * (this can be used to compute the compression ratio)
 */
uint32_t cpuCode::decompressor::getAccumulatedDecompressedDataSize(){
  return _accumDecompressedDataSize;
}

/*
 * Decompresses a dataframe. This function will decompress a dataframe in parallel and will call back with the decompressed data when completed.
 * The user should save the initialization vector dataframe and the elementCount to file himself. For dataframe index > 1 the user
 * should save the decompressed frame to persistent storage within the scope of the callback function. After the
 * return of the callback function the decompressed data will be deleted from memory and the pointers will no longer be valid.
 * @throws invalidInitializationException if the length of the dataframe vector does not match the length of the initialization vector
 */
void cpuCode::decompressor::decompressData(uint32_t elementCount, uint32_t chunkCount, uint32_t * chunkSizes, 
			uint32_t ** compressedResiduals, uint32_t ** compressedPrefixes, 
			void (*callBack)(uint32_t elementCount, uint32_t * decompressedData)){
  if (_decompressorIV == NULL || _decompressorIVLength != elementCount)
        throw invalidInitializationException();
  timer::tic();
  uint32_t NUMTHREADS = omp_get_max_threads();
  decompressionKernelArgs decompressionArgs[NUMTHREADS];
  #pragma omp parallel for 
  for (uint32_t dataBlockIndex = 0; dataBlockIndex < chunkCount; ++dataBlockIndex) {
    uint32_t index = dataBlockIndex%NUMTHREADS;
    decompressionArgs[index].chunkSize = chunkSizes[0];
    decompressionArgs[index].dataBlockSize = chunkSizes[dataBlockIndex];
    decompressionArgs[index].compressedPrefixes = compressedPrefixes[dataBlockIndex];
    decompressionArgs[index].compressedResiduals = compressedResiduals[dataBlockIndex];
    decompressionArgs[index].dataBlockIndex = dataBlockIndex;
    decompressionKernel((void *)&decompressionArgs[index]);
  }
  _decompressorAccumulatedTime += timer::toc();
  callBack(elementCount, _decompressorIV);
  _accumDecompressedDataSize += elementCount;
}
