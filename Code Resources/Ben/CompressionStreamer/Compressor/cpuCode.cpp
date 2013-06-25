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

void usedBitCount(uint32_t * data, int countData, int ignoreNumSignificantBits, int maxLeadingZeroCount, uint32_t * out){
  for (int i = 0; i < countData; ++i){
    int b = 31;
    int counter = 0; 
    for (b = 31 - ignoreNumSignificantBits; b >= (32 - maxLeadingZeroCount); --b){ 
      if (!(0x1 << b & ((int*)&data[0])[i]))
	++counter;
      else
	break;
    }
    out[i] = 32-(counter + ignoreNumSignificantBits);
  }
}

void createParallelPrefixSum(uint32_t * counts, uint32_t numElements) {
    //up-sweep:
    uint64_t upperBound = (uint64_t)log2(numElements)-1;
    for (uint64_t d = 0; d <= upperBound; ++d) {
        uint64_t twoTodPlus1 = (uint64_t)pow(2,d+1);
        #pragma omp parallel for shared(twoTodPlus1)
        for (uint64_t i = 0; i < numElements; i += twoTodPlus1) {
            counts[i + twoTodPlus1 - 1] += counts[i + twoTodPlus1/2 - 1];
        }
    }
    //clear:
    counts[numElements-1] = 0;
    //down-sweep:
    for (uint64_t d=upperBound; d >= 0; --d) {
        uint64_t twoTodPlus1 = (uint64_t)pow(2,d+1);
        #pragma omp parallel for shared(twoTodPlus1)
        for (uint64_t i = 0; i < numElements; i += twoTodPlus1) {
            uint32_t t = counts[i + twoTodPlus1/2 - 1];
            counts[i + twoTodPlus1/2 - 1] = counts[i + twoTodPlus1 - 1];
            counts[i + twoTodPlus1 - 1] += t;
        }
       if (d == 0) break;
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
void cpuCode::compressor::compressData(const float * data, uint64_t elementCount,
			  void (*callBack)(uint64_t compressedResidualsIntCount, uint32_t * compressedResiduals,
			    uint64_t compressedPrefixIntCount, uint32_t * compressedPrefixes)){
    if (_compressorIV == NULL || _compressorIVLength != elementCount)
        throw invalidInitializationException();

    uint8_t storageIndiceCapacity = 8*sizeof(uint32_t);
    
    /*
     * Create difference array
     */
    #pragma omp parallel for shared(_compressorIV)
    for (uint64_t i = 0; i < elementCount; ++i){
      _compressorIV[i] ^= ((uint32_t *)&data[0])[i];
    }
    
    /*
     * count leading zeros (ignore sign bit). Count up to 15 zeros (this needs 4 bits to store):
     */
    //the parallel prefix sum requires the size of the count array to be a power of 2:
    uint64_t indicesArraySize = (uint64_t)pow(2,ceil(log2(elementCount)));	
    uint32_t * arrIndexes = new uint32_t[indicesArraySize];
    //init array:
    for (int i = 0; i < indicesArraySize; ++i)
      arrIndexes[i] = 0;
    usedBitCount(_compressorIV, elementCount, 0, 15, arrIndexes);
    //save the first and last used bit counts, because they will be lost when the prefix sum is computed:
    uint8_t firstCount = arrIndexes[0];
    uint8_t lastCount = arrIndexes[elementCount - 1];

    /*
     * create storage for prefixes:
     */
    uint64_t sizeOfPrefixArray = ceil((elementCount * 4) / (float) storageIndiceCapacity);
    uint32_t * arrPrefix = new uint32_t[sizeOfPrefixArray];
    //init array:
    for (int i = 0; i < sizeOfPrefixArray; ++i)
      arrPrefix[i] = 0;
    /*
     * create compressed prefix array:
     */
    #pragma omp parallel for shared(arrPrefix)
    for (uint64_t i = 0; i < elementCount; ++i) {
        //store sign bit of the original data (shifted up) and then the 4 bits leading zero count (32 - used bits)
        uint8_t prefix = /*((((uint32_t*) &data[0])[i] >> storageIndiceCapacity - 1) << 4) |*/ (storageIndiceCapacity-arrIndexes[i]);
        //compact prefixes:
        int startingIndex = (i*4) / storageIndiceCapacity;
        int lshiftAmount = (storageIndiceCapacity - 4);
        int rshiftAmount = (i*4) % storageIndiceCapacity;
        int writtenBits = storageIndiceCapacity - fmax(lshiftAmount,rshiftAmount);
        #pragma omp atomic
        arrPrefix[startingIndex] |=
            ((prefix << lshiftAmount) >> rshiftAmount);
        if (4 - lshiftAmount - writtenBits > 0)
            #pragma omp atomic
            arrPrefix[startingIndex+1] |=
                (prefix << (lshiftAmount + writtenBits));
    }
    /*
     * create prefix sum (these are the starting (bit) indexes of the values):
     */
     createParallelPrefixSum(arrIndexes, indicesArraySize);
     
    /*
     * create storage for residuals:
     */
    uint64_t sizeOfResidualArray = ceil((arrIndexes[elementCount-1] + lastCount) / (float) storageIndiceCapacity);
    uint32_t * arrResiduals = new uint32_t[sizeOfResidualArray];
    //init array:
    for (int i = 0; i < sizeOfResidualArray; ++i)
      arrResiduals[i] = 0;
    /*
     * save residuals:
     */
    arrIndexes[0] = firstCount;
    //deal with the special case of the first element:
    {
        int startingIndex = 0;
        int lshiftAmount = (storageIndiceCapacity - firstCount);
	int rshiftAmount = 0 % storageIndiceCapacity;
        int writtenBits = storageIndiceCapacity - lshiftAmount - fmax(rshiftAmount-lshiftAmount,0);
        arrResiduals[startingIndex] = _compressorIV[0] << lshiftAmount;
    }
    //the inner bit is parallizable:
    #pragma omp parallel for shared(arrResiduals)
    for (int i=1; i < elementCount-1; ++i) {
        int startingIndex = arrIndexes[i] / storageIndiceCapacity;
        int lshiftAmount = (storageIndiceCapacity - (arrIndexes[i+1]-arrIndexes[i]));
        int rshiftAmount = arrIndexes[i] % storageIndiceCapacity;
        int writtenBits = storageIndiceCapacity - lshiftAmount - fmax(rshiftAmount-lshiftAmount,0);
        #pragma omp atomic
        arrResiduals[startingIndex] |=
            ( (_compressorIV[i] << lshiftAmount) >> rshiftAmount);
        if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)
            #pragma omp atomic
            arrResiduals[startingIndex+1] |=
                ( _compressorIV[i] << (lshiftAmount + writtenBits));
    }
  //deal with the special case of the last element
  if (elementCount > 1)
    {
        int startingIndex = arrIndexes[elementCount - 1] / storageIndiceCapacity;
        int lshiftAmount = (storageIndiceCapacity - lastCount);
        int rshiftAmount = arrIndexes[elementCount - 1] % storageIndiceCapacity;
        int writtenBits = storageIndiceCapacity - fmax(lshiftAmount,rshiftAmount);
        arrResiduals[startingIndex] |=
                             ( (_compressorIV[elementCount - 1] << lshiftAmount) >> rshiftAmount);
        if (storageIndiceCapacity - lshiftAmount - writtenBits > 0)
            arrResiduals[startingIndex+1] |=
                                   ( _compressorIV[elementCount - 1] << (lshiftAmount + writtenBits));
    }  
  arrIndexes[0] = 0;
  
  /*
   * Copy the current data to the IV memory for the next round of compression
   */
  memcpy(_compressorIV,data,elementCount*sizeof(float));
  
  /*
   * Finally call back to the caller with pointers to the compressed data & afterwards free the used data
   */
  callBack(sizeOfResidualArray,arrResiduals,sizeOfPrefixArray,arrPrefix);    
  delete[] arrIndexes;
  delete[] arrPrefix;
  delete[] arrResiduals;
}