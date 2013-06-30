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
uint32_t * _decompressorIV = NULL; 
uint64_t _decompressorIVLength = -1;
double _compressorAccumulatedTime = 0;
double _decompressorAccumulatedTime = 0;
const uint8_t storageIndiceCapacity = 8*sizeof(uint32_t);
const uint8_t bitCountForRepresentation = 5;

/*
 * Computes the binary logarithm of a 32 bit integer
 * Reference: Bit Twiddling Hacks by Sean Eron Anderson. Available at http://graphics.stanford.edu/~seander/bithacks.html
 * @params v a 32 bit unsigned integer
 */
inline uint32_t binaryLog32(uint32_t v){
    static const unsigned char LogTable256[256] =
    {
	#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
        -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
        LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
    };
    unsigned int t, tt; // temporaries
    if ((tt = v >> 16))
        return (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
    else
        return (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
}

inline int32_t imax( int32_t a, int32_t b )
{
    return a + ( ( b - a ) & ( (a - b) >> 31 ) );
}

/*
 * Computes the parallel prefix scan of an array
 * The first power of 2 indices can be computed in parallel as described by Blelloch (1990). The remaining indices are computed in serial.
 * @params counts is a 32bit unsigned int array pointer
 * @params numberElements is the number of elements in the passed array
 */
void createParallelPrefixSum(uint32_t * counts, uint32_t numElements) {
    uint32_t parallelLength = 1 << binaryLog32(numElements);
    if (parallelLength > 1) {
        uint32_t lastCount = counts[parallelLength-1];
        //up-sweep:
        uint32_t upperBound = (uint64_t)binaryLog32(parallelLength)-1;
        for (uint32_t d = 0; d <= upperBound; ++d) {
            uint32_t twoTodPlus1 = (1 << (d+1));
             #pragma omp parallel for shared(twoTodPlus1)
            for (uint32_t i = 0; i < parallelLength; i += twoTodPlus1) {
                counts[i + twoTodPlus1 - 1] += counts[i + (twoTodPlus1 >> 1) - 1];
            }
        }
        //clear:
        counts[parallelLength-1] = 0;
        //down-sweep:
        for (uint32_t d=upperBound; d >= 0; --d) {
            uint32_t twoTodPlus1 = (1 << (d+1));
             #pragma omp parallel for shared(twoTodPlus1)
            for (uint32_t i = 0; i < parallelLength; i += twoTodPlus1) {
                uint32_t t = counts[i + (twoTodPlus1 >> 1) - 1];
                counts[i + (twoTodPlus1 >> 1) - 1] = counts[i + twoTodPlus1 - 1];
                counts[i + twoTodPlus1 - 1] += t;
            }
            if (d == 0) break;
        }
        uint32_t serialLength = numElements - parallelLength;
        if (serialLength > 0) {
            int sum = counts[parallelLength-1]+lastCount;
            for (uint32_t i = parallelLength; i < parallelLength+serialLength; ++i) {
                int prevSum = sum;
                sum += counts[i];
                counts[i] = prevSum;
            }
        }
    } else {
        int serialLength = numElements - parallelLength;
        if (serialLength > 0) {
            int sum = counts[0];
            for (int i = parallelLength; i < serialLength; ++i) {
                int prevSum = sum;
                sum += counts[i];
                counts[i] = prevSum;
            }
            counts[0] = 0;
        }
    }
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
    delete[] _compressorIV;
  _compressorIV = new uint32_t[ivLength];
  memcpy(_compressorIV,iv,ivLength*sizeof(float));
  _compressorIVLength = ivLength;
  _compressorAccumulatedTime = 0;
}

/*
 * Releases resources held by the compressor
 */
void cpuCode::compressor::releaseResources(){
  if (_compressorIV != NULL){
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
 * Compresses a dataframe. This function will compress a dataframe in parallel and will call back with the compressed data when completed.
 * The user should save the initialization vector dataframe and the elementCount to file himself. For dataframe index > 1 the user
 * should save the compressed prefix and residual array to persistent storage within the scope of the callback function. After the
 * return of the callback function the compressed data will be deleted from memory and the pointers will no longer be valid.
 * @throws invalidInitializationException if the length of the dataframe vector does not match the length of the initialization vector
 */
void cpuCode::compressor::compressData(const float * data, uint32_t elementCount,
			  void (*callBack)(uint32_t elementCount, uint32_t compressedResidualsIntCount, uint32_t * compressedResiduals,
			    uint32_t compressedPrefixIntCount, uint32_t * compressedPrefixes)){
    Timer::tic();
    if (_compressorIV == NULL || _compressorIVLength != elementCount)
        throw invalidInitializationException();
    
    /*
     * create storage for counts and prefixes:
     */
    uint32_t sizeOfPrefixArray = (elementCount * bitCountForRepresentation) / storageIndiceCapacity +
      ((elementCount * bitCountForRepresentation) % storageIndiceCapacity != 0 ? 1 : 0);
    uint32_t * arrPrefix = new uint32_t[sizeOfPrefixArray](); //default initialize
    uint32_t * arrIndexes = new uint32_t[elementCount]; //no need to initialize we're going to override this in any case
    
    /*
     * Create difference array, count used bits (up to 31 leading zero bits) and save prefixes
     */
    #pragma omp parallel for shared(_compressorIV,arrIndexes,arrPrefix)
    for (uint32_t i = 0; i < elementCount; ++i){
      _compressorIV[i] ^= ((uint32_t *)&data[0])[i];
      register uint32_t lzc = imax(1,32-__builtin_clz (_compressorIV[i])); //this is an optimization for GCC compilers only, use fmax(1,(binaryLog32(_compressorIV[i]) + 1) &0x3f); otherwise
      arrIndexes[i] = lzc;
      //store the 5-bit leading zero count (32 - used bits)
      uint32_t prefix =  (storageIndiceCapacity-lzc);
      //compact prefixes:
      uint32_t startingIndex = (i*bitCountForRepresentation) / storageIndiceCapacity;
      uint8_t lshiftAmount = (storageIndiceCapacity - bitCountForRepresentation);
      uint8_t rshiftAmount = (i*bitCountForRepresentation) % storageIndiceCapacity;
      uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount - lshiftAmount,0);
      arrPrefix[startingIndex] |=
          ((prefix << lshiftAmount) >> rshiftAmount);
      if (storageIndiceCapacity - lshiftAmount - writtenBits > 0){
         arrPrefix[startingIndex+1] |=
            (prefix << (lshiftAmount + writtenBits));
      }
    }
    
    //save the first and last used bit counts, because they will be lost when the prefix sum is computed:
    uint8_t firstCount = arrIndexes[0];
    uint8_t lastCount = arrIndexes[elementCount - 1];

    /*
     * create prefix sum (these are the starting (bit) indexes of the values):
     */
     createParallelPrefixSum(arrIndexes, elementCount);

    /*
     * create storage for residuals:
     */
    uint32_t sizeOfResidualArray = (arrIndexes[elementCount-1] + lastCount) / storageIndiceCapacity + 
       ((arrIndexes[elementCount-1] + lastCount) % storageIndiceCapacity != 0 ? 1 : 0)+1;
    uint32_t * arrResiduals = new uint32_t[sizeOfResidualArray](); //default initialize
    
    /*
     * save residuals:
     */
    arrIndexes[0] = firstCount;
    //deal with the special case of the first element:
    {
        uint8_t lshiftAmount = (storageIndiceCapacity - firstCount);
        arrResiduals[0] = _compressorIV[0] << lshiftAmount;
    }
    //the inner bit is parallizable:
    #pragma omp parallel for shared(arrResiduals)
    for (uint32_t i=1; i < elementCount-1; ++i) {
        uint32_t index = arrIndexes[i];
	uint32_t ivElem = _compressorIV[i];
        uint32_t startingIndex = index / storageIndiceCapacity;
        uint8_t lshiftAmount = (storageIndiceCapacity - (arrIndexes[i+1]-index));
        uint8_t rshiftAmount = index % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
 	arrResiduals[startingIndex] |= ( (ivElem << lshiftAmount) >> rshiftAmount);
 	//if (storageIndiceCapacity - lshiftAmount - writtenBits > 0) //deliberitely made the array slightly larger to avoid branch divergence
 	arrResiduals[startingIndex+1] |= ( ivElem << (lshiftAmount + writtenBits - 1) << 1); 
    }
  //deal with the special case of the last element
  if (elementCount > 1)
    {
        uint32_t startingIndex = arrIndexes[elementCount - 1] / storageIndiceCapacity;
        uint8_t lshiftAmount = (storageIndiceCapacity - lastCount);
        uint8_t rshiftAmount = arrIndexes[elementCount - 1] % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount - lshiftAmount,0);
        arrResiduals[startingIndex] |=
                             ( (_compressorIV[elementCount - 1] << lshiftAmount) >> rshiftAmount);
//         if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)  //deliberitely made the array slightly larger to avoid branch divergence
        arrResiduals[startingIndex+1] |= ( _compressorIV[elementCount - 1] << (lshiftAmount + writtenBits - 1) << 1);
    }  
  
  /*
   * Copy the current data to the IV memory for the next round of compression
   */
   memcpy(_compressorIV,data,elementCount*sizeof(float));
  _compressorAccumulatedTime += Timer::toc();
  /*
   * Finally call back to the caller with pointers to the compressed data & afterwards free the used data
   */
  callBack(elementCount,sizeOfResidualArray-1,arrResiduals,sizeOfPrefixArray,arrPrefix);    
   delete[] arrIndexes;
   delete[] arrPrefix;
   delete[] arrResiduals;
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
 * Decompresses a dataframe. This function will decompress a dataframe in parallel and will call back with the decompressed data when completed.
 * The user should save the initialization vector dataframe and the elementCount to file himself. For dataframe index > 1 the user
 * should save the decompressed frame to persistent storage within the scope of the callback function. After the
 * return of the callback function the decompressed data will be deleted from memory and the pointers will no longer be valid.
 * @throws invalidInitializationException if the length of the dataframe vector does not match the length of the initialization vector
 */
void cpuCode::decompressor::decompressData(const uint32_t elementCount, const uint32_t* compressedResiduals, const uint32_t* compressedPrefixes, 
					   void (*callBack)(uint32_t elementCount, uint32_t * decompressedData)){
  Timer::tic();
  if (_decompressorIV == NULL || _decompressorIVLength != elementCount)
        throw invalidInitializationException();
  
  /*
   * create storage for counts and decompressed data:
   */
  uint32_t * arrIndexes = new uint32_t[elementCount]; //no need to initialize we're going to override this in any case
  
  /*
   * construct count array from prefix array 
   */
  #pragma omp parallel for shared(compressedPrefixes,arrIndexes)
  for (uint32_t i = 0; i < elementCount; ++i){
    uint32_t startingIndex = (i*bitCountForRepresentation) / storageIndiceCapacity;
    uint8_t lshiftAmount = (storageIndiceCapacity - bitCountForRepresentation);
    uint8_t rshiftAmount = (i*bitCountForRepresentation) % storageIndiceCapacity;
    uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - fmax(rshiftAmount - lshiftAmount,0);
    uint8_t prefix = ((compressedPrefixes[startingIndex] << rshiftAmount) >> lshiftAmount);
    if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)
         prefix |= (compressedPrefixes[startingIndex+1] >> (lshiftAmount + writtenBits-1) >> 1);
    arrIndexes[i] = 32 - prefix;
  }
  
  //save the first and last used bit counts, because they will be lost when the prefix sum is computed:
  uint8_t firstCount = arrIndexes[0];
  uint8_t lastCount = arrIndexes[elementCount - 1];
  
  /*
   * create prefix sum (these are the starting (bit) indexes of the values):
   */
  createParallelPrefixSum(arrIndexes, elementCount);

  /*
   * decompress residuals:
   */
  arrIndexes[0] = firstCount;
  //deal with the special case of the first element:
  {
      uint32_t startingIndex = 0;
      uint8_t lshiftAmount = (storageIndiceCapacity - firstCount);
      _decompressorIV[startingIndex] ^= (compressedResiduals[0] >> lshiftAmount);
  }
  //the inner bit is parallizable:
  #pragma omp parallel for shared(compressedResiduals)
  for (uint32_t i=1; i < elementCount-1; ++i) {
        uint32_t index = arrIndexes[i];
        uint32_t startingIndex = index / storageIndiceCapacity;
        uint8_t lshiftAmount = (storageIndiceCapacity - (arrIndexes[i+1]-index));
        uint8_t rshiftAmount = index % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount-lshiftAmount,0);
	register uint32_t residual = ( (compressedResiduals[startingIndex] << rshiftAmount) >> lshiftAmount);
 	if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)
 	  residual |= ( compressedResiduals[startingIndex+1] >> (lshiftAmount + writtenBits - 1) >> 1);
	_decompressorIV[i] ^= residual;
  }
  //deal with the special case of the last element
  if (elementCount > 1)
    {
        uint32_t startingIndex = arrIndexes[elementCount - 1] / storageIndiceCapacity;
        uint8_t lshiftAmount = (storageIndiceCapacity - lastCount);
        uint8_t rshiftAmount = arrIndexes[elementCount - 1] % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - imax(rshiftAmount - lshiftAmount,0);
        register uint32_t residual = ( (compressedResiduals[startingIndex] << rshiftAmount) >> lshiftAmount);
         if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)
 	  residual |= ( compressedResiduals[startingIndex+1] >> (lshiftAmount + writtenBits - 1) >> 1);
	_decompressorIV[elementCount - 1] ^= residual;
    }
  _decompressorAccumulatedTime += Timer::toc();
  callBack(elementCount,_decompressorIV);
  delete[] arrIndexes;
}
