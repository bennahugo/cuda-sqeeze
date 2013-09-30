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

/*
 * the following bittwiddled integer min,max functions assume INT_MIN <= x - y <= INT_MAX and are taken from
 * Sean Eron Anderson's Bit Twiddling Hacks. Available at http://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax
 */
const uint32_t BYTESPERINTMIN1 = sizeof(uint32_t) * sizeof(uint8_t) - 1;
inline int32_t imax( int32_t x, int32_t y )
{
    return x - ((x - y) & ((x - y) >> (BYTESPERINTMIN1)));
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


void compressionKernel(const float * data, uint32_t elementCount, uint32_t dataBlockIndex, 
			uint32_t chunkSize, uint32_t ** prefixStore, uint32_t ** residualStore, 
			uint32_t * prefixSizeStore, uint32_t * residualSizeStore, uint32_t * dataBlockSizes,
			uint32_t lowerBound, uint32_t elementsInDataBlock){
    uint32_t elementsDiv4 = elementsInDataBlock / 4;
    uint32_t remElements = elementsInDataBlock % 4;
    uint32_t uBound = elementsDiv4<<2;
    //create storage for counts and prefixes:
    uint32_t sizeOfPrefixArray = (elementsInDataBlock * bitCountForRepresentation) / storageIndiceCapacity +
                                 ((elementsInDataBlock * bitCountForRepresentation) % storageIndiceCapacity != 0);
    uint32_t * arrPrefix = (uint32_t*)_mm_malloc(sizeof(uint32_t)*sizeOfPrefixArray,16);
    memset(arrPrefix,0,sizeof(uint32_t)*sizeOfPrefixArray);
    uint32_t * arrResiduals = (uint32_t*)_mm_malloc(sizeof(uint32_t)*(elementsInDataBlock+1),16); //this padding actually waste less space than having a count array, +1 to avoid a branch later on when writing the remainder of the residuals
    memset(arrResiduals,0,sizeof(uint32_t)*elementsInDataBlock);

    //Create difference array, count used bits (up to 3 bytes of leading zeros) and save prefixes
    uint32_t lshiftAmountPrefixes = storageIndiceCapacity - bitCountForRepresentation;
    uint32_t accumulatedIndex = 0;
    __m128i lshiftAmounts = _mm_set1_epi32(lshiftAmountPrefixes);
    __m128i storageIndiceCapacities = _mm_set1_epi32(storageIndiceCapacity);
    __m128i bitCountsForRepresentation = _mm_set1_epi32(bitCountForRepresentation);
    __m128i indexOffsets = _mm_set_epi32(3,2,1,0);
    __m128i fives = _mm_set1_epi32(5);
    __m128i threes = _mm_set1_epi32(3);
    __m128i ones = _mm_set1_epi32(1);
    __m128i zeros = _mm_set1_epi32(0);
    __m128i minusones = _mm_set1_epi32(-1);
    Alignd(uint32_t startingIndexStore[4]);
    Alignd(uint32_t countsStore[4]);
    Alignd(uint32_t startingIndexPlus1Store[4]);
    Alignd(uint32_t elementsStore[4]);
    Alignd(uint32_t prefixesStore[4]);
    Alignd(uint32_t residlualsStore[4]);
    Alignd(uint32_t remainingBitsStore[4]);
    for (uint32_t i = 0; i < uBound; i+=4) {
	//compute xor and lzc:
        uint32_t index = i + lowerBound;
        
	__m128i elements = _mm_xor_si128(_mm_load_si128((__m128i*)(_compressorIV+index)),
					 _mm_load_si128((__m128i*)(data+index)));
	_mm_store_si128((__m128i*)(_compressorIV+index),elements);
	_mm_store_si128((__m128i*)elementsStore,elements);
	__m128i	prefixes = _mm_min_epi32(threes,
					 _mm_srl_epi32(_mm_set_epi32(fastLZC (elementsStore[3]),fastLZC (elementsStore[2]),
							fastLZC (elementsStore[1]),fastLZC (elementsStore[0])),threes));
	//save prefixes:
	__m128i iTimesBitCountsForRepresentation = _mm_mullo_epi32(_mm_add_epi32(_mm_set1_epi32(i),indexOffsets),bitCountsForRepresentation);
        __m128i startingIndexes = _mm_srl_epi32(iTimesBitCountsForRepresentation,fives);
	__m128i rshiftAmounts = _mm_sub_epi32(iTimesBitCountsForRepresentation,_mm_sll_epi32(startingIndexes,fives));
        uint32_t startingIndex = _mm_extract_epi32(startingIndexes,0);	//the starting indexes for up to 8 consequtive prefixes will be the same
        _mm_store_si128((__m128i*)prefixesStore,
			_mm_shl_epi32(_mm_sll_epi32(prefixes,lshiftAmounts),_mm_mullo_epi32(rshiftAmounts,minusones)));
	arrPrefix[startingIndex] |= prefixesStore[0] | prefixesStore[1] | prefixesStore[2] | prefixesStore[3];
        __m128i counts = _mm_sll_epi32(_mm_sub_epi32(_mm_set1_epi32(sizeof(uint32_t)),prefixes),threes);
        
	//save the residuals:
	_mm_store_si128((__m128i*)countsStore,counts);
	uint32_t counts0plus1 = countsStore[0]+countsStore[1];
	__m128i accumulatedIndexes = _mm_add_epi32(_mm_set1_epi32(accumulatedIndex),
						  _mm_set_epi32(counts0plus1+countsStore[2],
							       counts0plus1,countsStore[0],0));
        startingIndexes = _mm_srl_epi32(accumulatedIndexes,fives);
        __m128i residuallshiftAmounts = _mm_sub_epi32(storageIndiceCapacities,counts);
        rshiftAmounts = _mm_sub_epi32(accumulatedIndexes, _mm_sll_epi32(startingIndexes,fives));
	__m128i writtenBits = _mm_sub_epi32(_mm_sub_epi32(storageIndiceCapacities,residuallshiftAmounts),
									      _mm_max_epi32(
										_mm_sub_epi32(rshiftAmounts,residuallshiftAmounts),
										   zeros));
	_mm_store_si128((__m128i*)startingIndexStore,startingIndexes);
	__m128i residuals = _mm_shl_epi32(_mm_shl_epi32(elements,residuallshiftAmounts),_mm_mullo_epi32(rshiftAmounts,minusones));
	_mm_store_si128((__m128i*)residlualsStore,residuals);
        arrResiduals[startingIndexStore[0]] |= residlualsStore[0];
	arrResiduals[startingIndexStore[1]] |= residlualsStore[1];
	arrResiduals[startingIndexStore[2]] |= residlualsStore[2];
	arrResiduals[startingIndexStore[3]] |= residlualsStore[3];
	
	//store the remaining bits
	startingIndexes = _mm_add_epi32(startingIndexes,ones);
	_mm_store_si128((__m128i*)startingIndexPlus1Store,startingIndexes);
	__m128i remainingBits = _mm_sll_epi32(_mm_shl_epi32(elements,
							    _mm_sub_epi32(_mm_add_epi32(residuallshiftAmounts,writtenBits),ones)),
					      ones);
	_mm_store_si128((__m128i*)remainingBitsStore,remainingBits);
        arrResiduals[startingIndexPlus1Store[0]] |= remainingBitsStore[0];
        arrResiduals[startingIndexPlus1Store[1]] |= remainingBitsStore[1];
	arrResiduals[startingIndexPlus1Store[2]] |= remainingBitsStore[2];
	arrResiduals[startingIndexPlus1Store[3]] |= remainingBitsStore[3];
	
	accumulatedIndex = _mm_extract_epi32(accumulatedIndexes,3) + countsStore[3];
    }
    for (uint32_t r = 0; r < remElements; ++r) {
      uint32_t i = r + uBound;
      uint32_t index = i+lowerBound;
        //save the prefixes:
        uint32_t element = (_compressorIV[index] ^= ((uint32_t*)&(data[0]))[index]);
	uint32_t prefix0 = imin(3,fastLZC (element) >> 3);
        uint32_t iTimesBitCountForRepresentation = i*bitCountForRepresentation;
        uint32_t startingIndex = (iTimesBitCountForRepresentation) >> 5;
        uint32_t rshiftAmount = (iTimesBitCountForRepresentation) % storageIndiceCapacity;
        arrPrefix[startingIndex] |= ((prefix0 << lshiftAmountPrefixes) >> rshiftAmount);
        uint32_t count = ((sizeof(uint32_t)-prefix0) << 3);
        
	//save the residuals:
        startingIndex = accumulatedIndex >> 5;
        uint8_t lshiftAmount = (storageIndiceCapacity - count);
        rshiftAmount = accumulatedIndex % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
        element = _compressorIV[index]; //it seems after _lzcnt_u32 touches a memory location it is not optimized correctly this is a work arround
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
    
    
    //Copy the current data to the IV memory for the next round of compression
     memcpy(_compressorIV+lowerBound,data+lowerBound,elementsInDataBlock*sizeof(float));
    //the prefixes and residluals will be freed by the caller
}


void decompressionKernel(uint32_t chunkSize, uint32_t dataBlockSize, 
			  uint32_t * compressedPrefixes, uint32_t * compressedResiduals,
			  uint32_t dataBlockIndex,uint32_t lowerBound) {
    uint32_t elementsDiv4 = dataBlockSize / 4;
    uint32_t remElements = dataBlockSize % 4;
    uint32_t uBound = elementsDiv4<<2;

    uint32_t accumulatedIndex = 0;
    uint8_t lshiftAmount = (storageIndiceCapacity - bitCountForRepresentation);
    __m128i lshiftAmounts = _mm_set1_epi32(lshiftAmount);
    __m128i storageIndiceCapacities = _mm_set1_epi32(storageIndiceCapacity);
    __m128i bitCountsForRepresentation = _mm_set1_epi32(bitCountForRepresentation);
    __m128i indexOffsets = _mm_set_epi32(3,2,1,0);
    __m128i fives = _mm_set1_epi32(5);
    __m128i ones = _mm_set1_epi32(1);
    __m128i zeros = _mm_set1_epi32(0);
    __m128i minusones = _mm_set1_epi32(-1);
    Alignd(uint32_t startingIndexStore[4]);
    Alignd(uint32_t countsStore[4]);
    Alignd(uint32_t startingIndexPlus1Store[4]);
    for (uint32_t i = 0; i < uBound; i+=4) {
      //inflate prefix
      __m128i indexes = _mm_add_epi32(_mm_set1_epi32(i),indexOffsets);
      __m128i prefixIndexes = _mm_mullo_epi32(indexes,bitCountsForRepresentation);
      __m128i startingIndexes = _mm_srl_epi32(prefixIndexes,fives);  
      __m128i rshiftAmounts = _mm_sub_epi32(prefixIndexes,_mm_sll_epi32(startingIndexes,fives));
      uint32_t startingIndex = _mm_extract_epi32(startingIndexes,0);	//the starting indexes for up to 8 consequtive prefixes will be the same
      __m128i  deflatedPrefixes = _mm_set1_epi32(compressedPrefixes[startingIndex]);
      __m128i prefixes = _mm_srl_epi32(_mm_shl_epi32(deflatedPrefixes,rshiftAmounts),lshiftAmounts);
      __m128i counts = _mm_sub_epi32(storageIndiceCapacities,_mm_sll_epi32(prefixes,_mm_set1_epi32(3)));
      //inflate residual
      _mm_store_si128((__m128i*)countsStore,counts);
      uint32_t counts0plus1 = countsStore[0]+countsStore[1];
      __m128i accumulatedIndexes = _mm_add_epi32(_mm_set1_epi32(accumulatedIndex),
						  _mm_set_epi32(counts0plus1+countsStore[2],
							       counts0plus1,countsStore[0],0));
      startingIndexes = _mm_srl_epi32(accumulatedIndexes,fives);
      __m128i residuallshiftAmounts = _mm_sub_epi32(storageIndiceCapacities,counts);
      rshiftAmounts = _mm_sub_epi32(accumulatedIndexes,_mm_sll_epi32(startingIndexes,fives));
      __m128i writtenBits = _mm_sub_epi32(_mm_sub_epi32(storageIndiceCapacities,residuallshiftAmounts),
									      _mm_max_epi32(
										_mm_sub_epi32(rshiftAmounts,residuallshiftAmounts),
										   zeros));
      //grab the first few residaul bits:
      _mm_store_si128((__m128i*)startingIndexStore,startingIndexes);
      __m128i deflatedResiduals = _mm_set_epi32(compressedResiduals[startingIndexStore[3]],
						compressedResiduals[startingIndexStore[2]],
						compressedResiduals[startingIndexStore[1]],
						compressedResiduals[startingIndexStore[0]]);
      __m128i residuals = _mm_shl_epi32(_mm_shl_epi32(deflatedResiduals,rshiftAmounts),_mm_mullo_epi32(residuallshiftAmounts,minusones));
      //grab the remaining bits:
      __m128i startingIndexPlus1s = _mm_add_epi32(startingIndexes,
						  _mm_and_si128(_mm_cmpgt_epi32(_mm_sub_epi32(_mm_sub_epi32(storageIndiceCapacities,residuallshiftAmounts),
											      writtenBits),
										zeros),
								ones));
      
      _mm_store_si128((__m128i*)startingIndexPlus1Store,startingIndexPlus1s);
      __m128i nextShift = _mm_sub_epi32(_mm_add_epi32(residuallshiftAmounts,writtenBits),ones);
      
      __m128i deflatedRemainingBits = _mm_set_epi32(compressedResiduals[startingIndexPlus1Store[3]],
						compressedResiduals[startingIndexPlus1Store[2]],
						compressedResiduals[startingIndexPlus1Store[1]],
						compressedResiduals[startingIndexPlus1Store[0]]);
      __m128i remBits = _mm_srl_epi32(_mm_shl_epi32(deflatedRemainingBits,_mm_mullo_epi32(nextShift,minusones)),ones);
      uint32_t * _decompressorIVElem = _decompressorIV + lowerBound + i;
      _mm_store_si128((__m128i*)(_decompressorIVElem),_mm_xor_si128(_mm_load_si128((__m128i*)(_decompressorIVElem)),
								    _mm_or_si128(residuals,remBits)));
      accumulatedIndex = _mm_extract_epi32(accumulatedIndexes,3) + countsStore[3];
    }
    for (uint32_t r = 0; r < remElements; ++r) {
        uint32_t i = uBound + r;
        //inflate prefix
        uint32_t prefixIndex = i*bitCountForRepresentation;
        uint32_t startingIndex = prefixIndex >> 5;
        uint8_t rshiftAmount = prefixIndex % storageIndiceCapacity;
        uint8_t prefix = ((compressedPrefixes[startingIndex] << rshiftAmount) >> lshiftAmount);
        uint32_t count = storageIndiceCapacity - (prefix << 3);
        //inflate its associated residual
        startingIndex = accumulatedIndex >> 5;
        uint8_t residuallshiftAmount = (storageIndiceCapacity - count);
        rshiftAmount = accumulatedIndex % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - residuallshiftAmount - imax(rshiftAmount-residuallshiftAmount,0);
        register uint32_t residual = ( (compressedResiduals[startingIndex] << rshiftAmount) >> residuallshiftAmount);
        residual |=
            ( compressedResiduals[startingIndex+(storageIndiceCapacity - residuallshiftAmount - writtenBits > 0)] >> (residuallshiftAmount + writtenBits - 1) >> 1);
        _decompressorIV[lowerBound+i] ^= residual;
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
  #pragma omp parallel for 
  for (uint32_t dataBlockIndex = 0; dataBlockIndex < chunkCount; ++dataBlockIndex) {
    decompressionKernel(chunkSizes[0],chunkSizes[dataBlockIndex],compressedPrefixes[dataBlockIndex],
			compressedResiduals[dataBlockIndex],dataBlockIndex,dataBlockIndex*chunkSizes[0]);
  }
  _decompressorAccumulatedTime += timer::toc();
  callBack(elementCount, _decompressorIV);
  _accumDecompressedDataSize += elementCount;
}
