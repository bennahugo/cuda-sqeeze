/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2013  <copyright holder> <email>

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

static const unsigned char LogTable256[256] = 
{
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};
inline uint32_t binaryLog32(uint32_t v){
  unsigned int t, tt; // temporaries
  if (tt = v >> 16)
    return (t = tt >> 8) ? 24 + LogTable256[t] : 16 + LogTable256[tt];
  else
    return (t = v >> 8) ? 8 + LogTable256[t] : LogTable256[v];
}

void createParallelPrefixSum(uint32_t * counts, uint32_t numElements) {
    int parallelLength = 1 << binaryLog32(numElements);
    if (parallelLength > 1) {
        uint32_t lastCount = counts[parallelLength-1];
        //up-sweep:
        uint32_t upperBound = (uint64_t)binaryLog32(parallelLength)-1;
        for (uint32_t d = 0; d <= upperBound; ++d) {
            uint32_t twoTodPlus1 = 1 << d+1;
            #pragma omp parallel for shared(twoTodPlus1)
            for (uint64_t i = 0; i < parallelLength; i += twoTodPlus1) {
                counts[i + twoTodPlus1 - 1] += counts[i + (twoTodPlus1 >> 1) - 1];
            }
        }
        //clear:
        counts[parallelLength-1] = 0;
        //down-sweep:
        for (uint32_t d=upperBound; d >= 0; --d) {
            uint32_t twoTodPlus1 = 1 << d+1;
            #pragma omp parallel for shared(twoTodPlus1)
            for (uint32_t i = 0; i < parallelLength; i += twoTodPlus1) {
                uint32_t t = counts[i + (twoTodPlus1 >> 1) - 1];
                counts[i + (twoTodPlus1 >> 1) - 1] = counts[i + twoTodPlus1 - 1];
                counts[i + twoTodPlus1 - 1] += t;
            }
            if (d == 0) break;
        }
        int serialLength = numElements - parallelLength;
        if (serialLength > 0) {
            int sum = counts[parallelLength-1]+lastCount;
            for (int i = parallelLength; i < parallelLength+serialLength; ++i) {
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

void cpuCode::compressor::initCompressor(const float* iv, uint64_t ivLength){
  if (ivLength < 1)
    throw invalidInitializationException();
  if (_compressorIV != NULL)
    delete[] _compressorIV;
  _compressorIV = new uint32_t[ivLength];
  memcpy(_compressorIV,iv,ivLength*sizeof(float));
  _compressorIVLength = ivLength;
}

void cpuCode::compressor::releaseResources(){
  if (_compressorIV != NULL){
    delete[] _compressorIV;
    _compressorIV = NULL;
    _compressorIVLength = -1;
  }
}

void cpuCode::compressor::compressData(const float * data, uint32_t elementCount,
			  void (*callBack)(uint32_t compressedResidualsIntCount, uint32_t * compressedResiduals,
			    uint32_t compressedPrefixIntCount, uint32_t * compressedPrefixes)){
    if (_compressorIV == NULL || _compressorIVLength != elementCount)
        throw invalidInitializationException();

    uint32_t storageIndiceCapacity = 8*sizeof(uint32_t);
    
    /*
     * create storage for counts and prefixes:
     */
    uint8_t bitCountForRepresentation = 5;
    uint32_t sizeOfPrefixArray = (elementCount * bitCountForRepresentation) / storageIndiceCapacity +
      ((elementCount * bitCountForRepresentation) % storageIndiceCapacity != 0 ? 1 : 0);
    uint32_t * arrPrefix = new uint32_t[sizeOfPrefixArray](); //default initialize
    uint32_t * arrIndexes = new uint32_t[elementCount]; //no need to initialize we're going to override this in any case
    
    /*
     * Create difference array, count used bits (up to 15 leading zero bits) and save prefixes
     */
    #pragma omp parallel for shared(_compressorIV)
    for (uint32_t i = 0; i < elementCount; ++i){
      _compressorIV[i] ^= ((uint32_t *)&data[0])[i];
      arrIndexes[i] = fmax(1,(binaryLog32(_compressorIV[i]) + 1) & 0x3f); // &0x3f to mask out log2(0) + 1
      
      //store the 4-bit leading zero count (32 - used bits)
      uint8_t prefix =  (storageIndiceCapacity-arrIndexes[i]);
      //compact prefixes:
      uint32_t startingIndex = (i*bitCountForRepresentation) / storageIndiceCapacity;
      uint8_t lshiftAmount = (storageIndiceCapacity - bitCountForRepresentation);
      uint8_t rshiftAmount = (i*bitCountForRepresentation) % storageIndiceCapacity;
      uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - fmax(lshiftAmount - rshiftAmount,0);
      #pragma omp atomic
      arrPrefix[startingIndex] |=
          ((prefix << lshiftAmount) >> rshiftAmount);
      if (bitCountForRepresentation - lshiftAmount - writtenBits > 0)
         #pragma omp atomic
         arrPrefix[startingIndex+1] |=
            (prefix << (lshiftAmount + writtenBits-1) << 1);
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
       ((arrIndexes[elementCount-1] + lastCount) % storageIndiceCapacity != 0 ? 1 : 0) + 1; //avoid branch divergence later by using one more int
    uint32_t * arrResiduals = new uint32_t[sizeOfResidualArray](); //default initialize
    
    /*
     * save residuals:
     */
    arrIndexes[0] = firstCount;
    //deal with the special case of the first element:
    {
        uint32_t startingIndex = 0;
        uint8_t lshiftAmount = (storageIndiceCapacity - firstCount);
	uint8_t rshiftAmount = 0 % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - fmax(rshiftAmount-lshiftAmount,0);
        arrResiduals[startingIndex] = _compressorIV[0] << lshiftAmount;
    }
    //the inner bit is parallizable:
    #pragma omp parallel for shared(arrResiduals)
    for (uint32_t i=1; i < elementCount-1; ++i) {
        uint32_t index = arrIndexes[i];
	uint32_t ivElem = _compressorIV[i];
        uint32_t startingIndex = index / storageIndiceCapacity;
        uint8_t lshiftAmount = (storageIndiceCapacity - (arrIndexes[i+1]-index));
        uint8_t rshiftAmount = index % storageIndiceCapacity;
        //uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - fmax(rshiftAmount-lshiftAmount,0);
	uint64_t* resElem = (uint64_t*) (arrResiduals + startingIndex);
	#pragma omp atomic
	resElem[0] |= ( (ivElem << lshiftAmount) >> rshiftAmount);
//         #pragma omp atomic
// 	arrResiduals[startingIndex] |=
//             ( (ivElem << lshiftAmount) >> rshiftAmount);
//         
//         #pragma omp atomic
//         arrResiduals[startingIndex+1] |=
//             ( ivElem << (lshiftAmount + writtenBits - 1) << 1);
    }
  //deal with the special case of the last element
  if (elementCount > 1)
    {
        uint32_t startingIndex = arrIndexes[elementCount - 1] / storageIndiceCapacity;
        uint8_t lshiftAmount = (storageIndiceCapacity - lastCount);
        uint8_t rshiftAmount = arrIndexes[elementCount - 1] % storageIndiceCapacity;
        uint8_t writtenBits = storageIndiceCapacity - lshiftAmount - fmax(rshiftAmount - lshiftAmount,0);
        arrResiduals[startingIndex] |=
                             ( (_compressorIV[elementCount - 1] << lshiftAmount) >> rshiftAmount);
        if (storageIndiceCapacity - lshiftAmount - writtenBits > 0){
            arrResiduals[startingIndex+1] |=
                                   ( _compressorIV[elementCount - 1] << (lshiftAmount + writtenBits));
	};
    }  
  arrIndexes[0] = 0;
  
  /*
   * Copy the current data to the IV memory for the next round of compression
   */
   memcpy(_compressorIV,data,elementCount*sizeof(float));
  
  /*
   * Finally call back to the caller with pointers to the compressed data & afterwards free the used data
   */
  callBack(sizeOfResidualArray-1,arrResiduals,sizeOfPrefixArray,arrPrefix);    
   delete[] arrIndexes;
   delete[] arrPrefix;
   delete[] arrResiduals;
}

void cpuCode::decompressor::initDecompressor(const float* iv, uint64_t ivLength){
  if (ivLength < 1)
    throw invalidInitializationException();
  if (_decompressorIV != NULL)
    delete[] _decompressorIV;
  _decompressorIV = new uint32_t[ivLength];
  memcpy(_decompressorIV,iv,ivLength*sizeof(float));
  _decompressorIVLength = ivLength;
}

void cpuCode::decompressor::releaseResources(){
  if (_decompressorIV != NULL){
    delete[] _decompressorIV;
    _decompressorIV = NULL;
    _decompressorIVLength = -1;
  }
}