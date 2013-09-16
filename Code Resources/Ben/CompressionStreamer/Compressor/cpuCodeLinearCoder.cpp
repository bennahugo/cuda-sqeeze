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


#include "cpuCodeLinearCoder.h"
#define PREDICTOR_ORDER 2
uint32_t ** _compressorIV = NULL; 
uint64_t _compressorIVLength = -1;
uint32_t _accumCompressedDataSize = 0;
uint32_t ** _decompressorIV = NULL; 
uint64_t _decompressorIVLength = -1;
double _compressorAccumulatedTime = 0;
double _decompressorAccumulatedTime = 0;
uint32_t _accumDecompressedDataSize = 0;
const uint8_t storageIndiceCapacity = 8*sizeof(uint32_t);
const uint8_t bitCountForRepresentation = 3;

// Define macro for aligning data
#ifdef _MSC_VER
// If Microsoft compiler
#define Alignd(X) __declspec(align(16)) X
#else
// Gnu compiler, etc.
#define Alignd(X) X __attribute__((aligned(16)))
#endif

/*
 * the following bittwiddled integer min,max functions assume INT_MIN <= x - y <= INT_MAX and are taken from
 * Sean Eron Anderson's Bit Twiddling Hacks. Available at http://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax
 */
const uint32_t BYTESPERINTMIN1 = sizeof(uint32_t) * sizeof(uint8_t) - 1;
inline int32_t imax( int32_t x, int32_t y )
{
    return x ^ ((x ^ y) & -(x < y)); // max(x, y)
}
inline int32_t imin( int32_t x, int32_t y )
{
    return y ^ ((x ^ y) & -(x < y)); // min(x, y)
}

// Returns the number of leading 0-bits in x, starting at the most significant bit position.
// If x is zero, the result is undefined.
uint32_t lzc(register uint32_t x)
{
  // This uses a binary search (counting down) algorithm from Hacker's Delight.
   register uint32_t y;
   uint32_t n = 32;
   y = x >>16;  if (y != 0) {n = n -16;  x = y;}
   y = x >> 8;  if (y != 0) {n = n - 8;  x = y;}
   y = x >> 4;  if (y != 0) {n = n - 4;  x = y;}
   y = x >> 2;  if (y != 0) {n = n - 2;  x = y;}
   y = x >> 1;  if (y != 0) return n - 2;
   return n - x;
}

inline uint32_t fastLZC (register uint32_t x){
	uint32_t result;
 	asm("LZCNT %0,%1" : "=r" (result) : "r" (x));	
 	return result;
}

inline int32_t compressorParallelogramPredictor(uint32_t index){
#if (PREDICTOR_ORDER != 3)
  std::cerr << "PARALLELOGRAM PREDICTOR CAN ONLY DEFINED FOR A 3 ELEMENT SCHEME" << std::endl;
  exit(1);
#endif  
  int32_t P = 0;
  for (uint32_t j = 0; j < PREDICTOR_ORDER; ++j)
    P += (-2*(1-(int)j%2)+1)*((int32_t**)_compressorIV)[j][index];
  return P;
}
inline int32_t decompressorParallelogramPredictor(uint32_t index){
#if (PREDICTOR_ORDER != 3)
  std::cerr << "PARALLELOGRAM PREDICTOR CAN ONLY DEFINED FOR A 3 ELEMENT SCHEME" << std::endl;
  exit(1);
#endif  
  int32_t P = 0;
  for (uint32_t j = 0; j < PREDICTOR_ORDER; ++j)
    P += (-2*(1-(int)j%2)+1)*((int32_t**)_decompressorIV)[j][index];
  return P;
}
inline int32_t compressorMeanPredictor(uint32_t index){
  int32_t P = 0;
  for (uint32_t j = 0; j < PREDICTOR_ORDER; ++j)
    P += ((int32_t**)_compressorIV)[j][index];
  return P/PREDICTOR_ORDER;
}
inline int32_t decompressorMeanPredictor(uint32_t index){
  int32_t P = 0;
  for (uint32_t j = 0; j < PREDICTOR_ORDER; ++j)
    P += ((int32_t**)_decompressorIV)[j][index];
  return P/PREDICTOR_ORDER;
}
inline int32_t compressorLagrangePredictor(uint32_t index){
   uint32_t fStar = PREDICTOR_ORDER + 1;
   int32_t y = 0;
   for (uint32_t i = 1; i <= PREDICTOR_ORDER; ++i){
       int32_t L = 1;
       int32_t nom = 1;
       int32_t denum = 1;
       for (uint32_t j = 1; j <= i-1; ++j){
	  nom *= (fStar-j);
	  denum *= (i-j);
       }
       for (uint32_t j = i+1; j <= PREDICTOR_ORDER; ++j){
	  nom *= (fStar-j);
	  denum *= (i-j);
       }
       y += ((int32_t**)_compressorIV)[i-1][index]*(nom/denum);
   }
    return y;
}
inline int32_t decompressorLagrangePredictor(uint32_t index){
   uint32_t fStar = PREDICTOR_ORDER + 1;
   int32_t y = 0;
   for (uint32_t i = 1; i <= PREDICTOR_ORDER; ++i){
       int32_t nom = 1;
       int32_t denum = 1;
       for (uint32_t j = 1; j <= i-1; ++j){
	  nom *= (fStar-j);
	  denum *= (i-j);
       }
       for (uint32_t j = i+1; j <= PREDICTOR_ORDER; ++j){
	  nom *= (fStar-j);
	  denum *= (i-j);
       }
       y += ((int32_t**)_decompressorIV)[i-1][index]*(nom/denum);
   }
    return y;
}
/*---------------------------------------------------------------------------
   Fast Pivot-based median (based on a kth-largest algorithm suggested by Niklaus Wirth)
   The algorithm has been shown to have linear average complexity (which is faster than a
   naive nlog(n) performance using sorting.
    Reference:
                  Author: Wirth, Niklaus 
                   Title: Algorithms + data structures = programs 
               Publisher: Englewood Cliffs: Prentice-Hall, 1976 
    Physical description: 366 p. 
                  Series: Prentice-Hall Series in Automatic Computation 
 ---------------------------------------------------------------------------*/
int32_t pivotMedian(int32_t *data, int n) {
    int i, j, l, m, k = n/2-1;
    int32_t x, s;

    l=0;
    m=n-1;
    while(l<m) {
        x=data[k];
        i=l;
        j=m;
        do {
            while(data[i]<x) i++;
            while(x<data[j]) j--;
            if(i<=j) {
                s=data[i];
                data[i]=data[j];
                data[j]=s;
                i++;
                j--;
            }
        } while(i<=j);
        if(j<k) l=i;
        if(k<i) m=j;
    }
    return(data[k]);
}
inline int32_t compressorMedianPredictor(uint32_t index){
  int32_t* prev = new int32_t[PREDICTOR_ORDER];
  for (uint32_t j = 0; j < PREDICTOR_ORDER; ++j)
    prev[j] = ((int32_t**)_compressorIV)[j][index];
  int32_t P = pivotMedian(prev,PREDICTOR_ORDER);
  delete[] prev;
  return P;
}
inline int32_t decompressorMedianPredictor(uint32_t index){
  int32_t* prev = new int32_t[PREDICTOR_ORDER];
  for (uint32_t j = 0; j < PREDICTOR_ORDER; ++j)
    prev[j] = ((int32_t**)_decompressorIV)[j][index];
  int32_t P = pivotMedian(prev,PREDICTOR_ORDER);
  delete[] prev;
  return P;
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
  if (_compressorIV != NULL){
    for (uint32_t i = 0; i < PREDICTOR_ORDER; ++i)
      _mm_free(_compressorIV[i]);
    delete[] _compressorIV;
  }
  _compressorIV = new uint32_t*[PREDICTOR_ORDER];
  for (uint32_t i = 0; i < PREDICTOR_ORDER; ++i){
    _compressorIV[i] = (uint32_t*)_mm_malloc(sizeof(uint32_t)*ivLength,16);
    memcpy(_compressorIV[i],iv,ivLength*sizeof(uint32_t));
  }
  _compressorIVLength = ivLength;
  _compressorAccumulatedTime = 0;
  _accumCompressedDataSize = ivLength+1;
}

/*
 * Releases resources held by the compressor
 */
void cpuCode::compressor::releaseResources(){
  if (_compressorIV != NULL){
    for (uint32_t i = 0; i < PREDICTOR_ORDER; ++i)
      _mm_free(_compressorIV[i]);
    delete[] _compressorIV;
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

void compressionKernel(const float * data, uint32_t elementCount, uint32_t dataBlockIndex, 
			uint32_t chunkSize, uint32_t ** prefixStore, uint32_t ** residualStore, 
			uint32_t * prefixSizeStore, uint32_t * residualSizeStore, uint32_t * dataBlockSizes,
			uint32_t lowerBound, uint32_t elementsInDataBlock){

    //create storage for counts and prefixes:
    uint32_t sizeOfPrefixArray = (elementsInDataBlock * bitCountForRepresentation) / storageIndiceCapacity +
                                 ((elementsInDataBlock * bitCountForRepresentation) % storageIndiceCapacity != 0);
    uint32_t * arrPrefix = (uint32_t*)_mm_malloc(sizeof(uint32_t)*(sizeOfPrefixArray+1),16);
    memset(arrPrefix,0,sizeof(uint32_t)*(sizeOfPrefixArray+1));
    uint32_t * arrResiduals = (uint32_t*)_mm_malloc(sizeof(uint32_t)*(elementsInDataBlock+1),16); //this padding actually waste less space than having a count array, +1 to avoid a branch later on when writing the remainder of the residuals
    memset(arrResiduals,0,sizeof(uint32_t)*(elementsInDataBlock+1));

    //Create difference array, count used bits (up to 3 bytes of leading zeros) and save prefixes
    uint32_t lshiftAmountPrefixes = storageIndiceCapacity - bitCountForRepresentation;
    uint32_t accumulatedIndex = 0;
    for (uint32_t i = 0; i < elementsInDataBlock; ++i) {
	uint32_t index = i+lowerBound;
        //save the prefixes:
	int32_t P = compressorMedianPredictor(index);
	for (uint32_t j = 1; j < PREDICTOR_ORDER;++j)
	  _compressorIV[j-1][index] = _compressorIV[j][index];
	_compressorIV[PREDICTOR_ORDER-1][index] = *((int32_t*)&data[index]); 
	uint32_t element = (*((int32_t*)&data[index]) ^ P);
 	uint32_t sign = (element >> 31) << 2;
	uint32_t prefix0 = imin(3,lzc (element & 0x7FFFFFFF) >> 3);
	//element = _compressorIV[PREDICTOR_ORDER-1][index]; //it seems after _lzcnt_u32 touches a memory location it is not optimized correctly this is a work arround
	uint32_t count = ((sizeof(uint32_t)-prefix0) << 3);
	prefix0 |= sign;
        uint32_t iTimesBitCountForRepresentation = i * bitCountForRepresentation;
        uint32_t startingIndex = (iTimesBitCountForRepresentation) >> 5;
        uint32_t rshiftAmount = (iTimesBitCountForRepresentation) % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmountPrefixes - imax(rshiftAmount-lshiftAmountPrefixes,0);
	arrPrefix[startingIndex] |= ((prefix0 << lshiftAmountPrefixes) >> rshiftAmount);
	arrPrefix[startingIndex+1] |= (prefix0 << (lshiftAmountPrefixes + writtenBits - 1) << 1);
	
	//save the residuals:
        startingIndex = accumulatedIndex >> 5;
        uint8_t lshiftAmount = (storageIndiceCapacity - count);
        rshiftAmount = accumulatedIndex % storageIndiceCapacity;
        writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
        arrResiduals[startingIndex] |= ( (element << lshiftAmount) >> rshiftAmount);
        arrResiduals[startingIndex+1] |= (element << (lshiftAmount + writtenBits - 1) << 1);
        accumulatedIndex += count;
    }
 
    //calculate storage space used by residuals:
    uint32_t sizeOfResidualArray = accumulatedIndex / storageIndiceCapacity +
                          (accumulatedIndex % storageIndiceCapacity != 0);

    //Store pointers to the current prefixes and residuals
    residualStore[dataBlockIndex] = arrResiduals;
    prefixStore[dataBlockIndex] = arrPrefix;
    residualSizeStore[dataBlockIndex] = sizeOfResidualArray;
    prefixSizeStore[dataBlockIndex] = sizeOfPrefixArray; 
    dataBlockSizes[dataBlockIndex] = elementsInDataBlock;
    
    //the prefixes and residluals will be freed by the caller
}

void decompressionKernel(uint32_t chunkSize, uint32_t dataBlockSize, 
			  uint32_t * compressedPrefixes, uint32_t * compressedResiduals,
			  uint32_t dataBlockIndex,uint32_t lowerBound) {
    uint32_t accumulatedIndex = 0;
    uint8_t lshiftAmount = (storageIndiceCapacity - bitCountForRepresentation);
    for (uint32_t i = 0; i < dataBlockSize; ++i) {
	uint32_t index = lowerBound+i;
	//inflate prefix
	uint32_t prefixIndex = i*bitCountForRepresentation;
        uint32_t startingIndex = prefixIndex >> 5;
        uint8_t rshiftAmount = prefixIndex % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
	uint8_t prefix = ((compressedPrefixes[startingIndex] << rshiftAmount) >> lshiftAmount);
 	prefix |= ( compressedPrefixes[startingIndex+(storageIndiceCapacity - lshiftAmount - writtenBits > 0)] >> (lshiftAmount + writtenBits - 1) >> 1);
	uint32_t sign = (prefix >> 2) << 31;
	prefix &= 0x03;
        uint32_t count = storageIndiceCapacity - (prefix << 3);
	
	//inflate its associated residual
	startingIndex = accumulatedIndex >> 5;
        uint8_t residuallshiftAmount = (storageIndiceCapacity - count);
        rshiftAmount = accumulatedIndex % storageIndiceCapacity;
        writtenBits = storageIndiceCapacity - residuallshiftAmount - imax(rshiftAmount-residuallshiftAmount,0);
        uint32_t element = ( (compressedResiduals[startingIndex] << rshiftAmount) >> residuallshiftAmount) 
	  | ( compressedResiduals[startingIndex+(storageIndiceCapacity - residuallshiftAmount - writtenBits > 0)] >> (residuallshiftAmount + writtenBits - 1) >> 1)
	  | sign;
	uint32_t residual = *((uint32_t*)&element);
	int32_t P = decompressorMedianPredictor(index);
	//P += _decompressorIV[PREDICTOR_ORDER-1][index];
	for (uint32_t j = 1; j < PREDICTOR_ORDER;++j)
	  _decompressorIV[j-1][index] = _decompressorIV[j][index];
	_decompressorIV[PREDICTOR_ORDER-1][index] = P ^ residual;
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

void cpuCode::compressor::compressData(const float * data, uint32_t elementCount,
			  void (*callBack)(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, uint32_t chunkCount, uint32_t * chunkSizes)){
    if (_compressorIV == NULL || _compressorIVLength != elementCount)
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
      compressionKernel(data,elementCount,dataBlockIndex,chunkSize,prefixStore,
			residlualStore,prefixSizesStore,residualSizesStore,chunkSizes,
			dataBlockIndex*chunkSize,
			(((dataBlockIndex + 1)*chunkSize <= elementCount) ? chunkSize : chunkSize-((dataBlockIndex + 1)*chunkSize-elementCount)));
    }
      compressionKernel(data,elementCount,NUMTHREADS,chunkSize,prefixStore,
			residlualStore,prefixSizesStore,residualSizesStore,chunkSizes,
			NUMTHREADS*chunkSize,
			(((NUMTHREADS + 1)*chunkSize <= elementCount) ? chunkSize : chunkSize-((NUMTHREADS + 1)*chunkSize-elementCount)));
    _compressorAccumulatedTime += timer::toc();
    //Now do the callback and free all resources afterwards except the IV:
    for (uint32_t i = 0; i < numStores; ++i){
      _accumCompressedDataSize += residualSizesStore[i] + prefixSizesStore[i] + 1;
    }
    callBack(elementCount,residualSizesStore,residlualStore,prefixSizesStore,prefixStore,numStores,chunkSizes);
    for (uint32_t i = 0; i < numStores; ++i){
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
  if (_decompressorIV != NULL){
    for (uint32_t i = 0; i < PREDICTOR_ORDER; ++i)
      _mm_free(_decompressorIV[i]);
    delete[] _decompressorIV;
  }
  _decompressorIV = new uint32_t*[PREDICTOR_ORDER];
  for (uint32_t i = 0; i < PREDICTOR_ORDER; ++i){
    _decompressorIV[i] = (uint32_t*)_mm_malloc(sizeof(uint32_t)*ivLength,16);
    memcpy(_decompressorIV[i],iv,ivLength*sizeof(uint32_t));
  }
  _decompressorIVLength = ivLength;
  _decompressorAccumulatedTime = 0;
  _accumDecompressedDataSize = ivLength;
}

/*
 * Releases resources held by the decompressor
 */
void cpuCode::decompressor::releaseResources(){
  if (_decompressorIV != NULL){
    for (uint32_t i = 0; i < PREDICTOR_ORDER; ++i)
      _mm_free(_decompressorIV[i]);
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
  #pragma omp parallel for 
  for (uint32_t dataBlockIndex = 0; dataBlockIndex < chunkCount; ++dataBlockIndex) {
    decompressionKernel(chunkSizes[0],chunkSizes[dataBlockIndex],compressedPrefixes[dataBlockIndex],
			compressedResiduals[dataBlockIndex],dataBlockIndex,dataBlockIndex*chunkSizes[0]);
  }
  _decompressorAccumulatedTime += timer::toc();
  callBack(elementCount, (uint32_t*)_decompressorIV[PREDICTOR_ORDER-1]);
  _accumDecompressedDataSize += elementCount;
}