#include <iostream>
#include <string>
#include "AstroReader/file.h"
#include "AstroReader/stride.h"
#include "AstroReader/stridefactory.h"
#include "streamcollector.h"
#include <cstring>
#include <math.h>

#define FILENAME "/media/OS/SKA_DATA/kat7_data/1369858495.h5"
void printBinaryRepresentation(void * data, int sizeInBytes);
void xorArray(void * out, void * data, void * data2, int sizeInBytes);
void processStride(const astroReader::stride & data);
void createBitCountArray(float * data, int countData, int ignoreNumSignificantBits, int * out);
void createPrefixSumArray(int * counts, int numElements, int * out);
void bitsPack(float * data,int firstElementCount, int lastElementCount, 
	      int * startingIndexes, int * out, int countData, int countOut);
int numberOfBytesNeededToCompress(int * startingIndexes,int countData, int lastCount);
void createParallelPrefixSum(int * counts, int numElements);

int main(int argc, char **argv) {
    using namespace std;
    /*astroReader::file f(string(FILENAME));
    cout << "File dimensions: ";
    for (int i = 0; i < f.getDimensionCount(); ++i)
      cout << f.getDimensionSize(i) << ((i == f.getDimensionCount() -1) ? "\n" : " x ");
    streamCollector::chunkedRead(f,1024,2,processStride);
    */
    
    //astroReader::stride data = astroReader::strideFactory::createStride(f,80,0,50,0,10,0); 
    //data.print();
    /*complexPair<float> e1 = data.getElement(0,0,0);
    complexPair<float> e2 = data.getElement(1,0,0);
    cout << e1.r << "   " << e2.r << endl;
    float t = 0;
    xorArray(&t,&e1.r,&e2.r,sizeof(float));
    printBinaryRepresentation(&e1.r, sizeof(float));
    printBinaryRepresentation(&e2.r, sizeof(float));
    printBinaryRepresentation(&t, sizeof(float));*/
    
    /*for (int t = 0; t < data.getMaxTimestampIndex() - data.getMinTimestampIndex(); ++t)
      for (int f = 0; f <= data.getMaxFreqIndex() - data.getMinFreqIndex(); ++f)
	for (int c = 0; c <= data.getMaxCorrelationPairIndex() - data.getMinCorrelationPairIndex(); ++c){
	  complexPair<float> e1 = data.getElement(t,f,c);
	  complexPair<float> e2 = data.getElement(t+1,f,c);
	  float t = 0;
	  float t1 = 0;
	  xorArray(&t,&e1.r,&e2.r,sizeof(float));
	  xorArray(&t1,&e1.i,&e2.i,sizeof(float));
	  printBinaryRepresentation(&t, sizeof(float));
	  printBinaryRepresentation(&t1, sizeof(float));
	}*/
    int testCount = 10;
    float data[10] = {0,0,0,0,2.532f,2.532f,2.532f,2.532f,2.532f,2.532f};
    int * counts = new int[testCount];
    /*for (int i = 0; i < testCount; ++i){
      data[i] = 2.532f;
       if (i <5)
 	xorArray(&data[i],&data[i],&data[i],sizeof(float));
      cout << "Element " << i << ": ";
      printBinaryRepresentation(&data[i],sizeof(float));
    }*/
    createBitCountArray(data,testCount,1,counts);
    int firstCount = counts[0];
    int lastCount = counts[testCount -1];
    int * counts2 = new int[(int)pow(2,ceil(log2(testCount)))];
    createBitCountArray(data,testCount,1,counts2);
    
    createParallelPrefixSum(counts2, (int)pow(2,ceil(log2(testCount))));
    createPrefixSumArray(counts, testCount, counts);
    cout << "Prefix sum array :[";
    for (int i = 0; i < testCount; ++i){
      cout << counts[i] << ",";
    }
    cout << ']' << endl;
    cout << "Parallel prefix sum array :[";
    for (int i = 0; i < testCount; ++i){
      cout << counts2[i] << ",";
    }
    cout << ']' << endl;
    
    int compressedDataCount = numberOfBytesNeededToCompress(counts,testCount, lastCount);
    int * compressedData = new int[compressedDataCount];
   
    bitsPack(&data[0],firstCount, lastCount, &counts[0],compressedData,testCount,compressedDataCount);
    cout << "COMPRESSED DATA:" << endl;
    for (int i = 0; i < compressedDataCount; ++i)
      printBinaryRepresentation(&(compressedData[i]),sizeof(float));
    delete[] compressedData;
    delete[] counts;
    delete[] counts2;
    return 0;
}
void processStride(const astroReader::stride & data){
  //TODO: DO STUFF WITH THE STRIDES OF READ DATA
  //data.print();
  /*for (int t = 0; t < data.getMaxTimestampIndex() - data.getMinTimestampIndex(); ++t)
      for (int f = 0; f <= data.getMaxFreqIndex() - data.getMinFreqIndex(); ++f)
	  for (int c = 0; c <= data.getMaxCorrelationPairIndex() - data.getMinCorrelationPairIndex(); ++c){
	   
	  }*/
}
void xorArray(void * out, void * data, void * data2, int sizeInBytes){
  char * tout = (char *)out;
  char * tdata = (char *)data;
  char * tdata2 = (char *)data2;
  for (int i = 0; i < sizeInBytes; ++i)
    tout[i] = tdata[i] xor tdata2[i];
}
void printBinaryRepresentation(void * data, int sizeInBytes){
  using namespace std;
  char * temp = (char *)data;
  for (int i = sizeInBytes - 1; i >= 0; --i){
#pragma loop unroll
    for (int b = 7; b >= 0; --b)  
      cout << (0x1 << b & temp[i] ? '1' : '0');
    cout << ' ';
  }
  cout << endl;
}
void createBitCountArray(float * data, int countData, int ignoreNumSignificantBits, int * out){
  for (int i = 0; i < countData; ++i){
    int b = 31;
    int counter = 0; 
    for (b = 31 - ignoreNumSignificantBits; b >= 0; --b){ 
      if (!(0x1 << b & ((int*)&data[0])[i]))
	++counter;
      else
	break;
    }
    out[i] = 32 - (counter + ignoreNumSignificantBits);
  }
}
void createPrefixSumArray(int * counts, int numElements, int * out){
  int sum = counts[0];
  for (int i = 1; i < numElements; ++i){
    int prevSum = sum;
    sum += counts[i];
    out[i] = prevSum;
  }
  out[0] = 0;
}
void createParallelPrefixSum(int * counts, int numElements){
  //up-sweep:
  int upperBound = (int)log2(numElements);
  for (int d = 1; d <= upperBound; ++d){
    int twoTodPlus1 = (int)pow(2,d);
    //parallel:
    for (int i = 0; i < numElements; i += twoTodPlus1){
      counts[i + twoTodPlus1 - 1] += counts[i + twoTodPlus1/2 - 1];  
    }
  }
  //clear:
  counts[numElements-1] = 0;
  //down-sweep:
  for (int d=upperBound; d >= 0; --d){
    int twoTodPlus1 = (int)pow(2,d+1);
    //parallel:
    for (int i = 0; i < numElements; i += twoTodPlus1){
      int t = counts[i + twoTodPlus1/2 - 1];
      counts[i + twoTodPlus1/2 - 1] = counts[i + twoTodPlus1 - 1];
      counts[i + twoTodPlus1 - 1] += t;
    }
  }
}
int numberOfBytesNeededToCompress(int * startingIndexes,int countData, int lastCount){
  return (int)ceil(ceil((startingIndexes[countData-1] + lastCount) / 8.0f) / 4.0f);
}
void bitsPack(float * data,int firstElementCount, int lastElementCount, int * startingIndexes, int * out, int countData, int countOut){  
  startingIndexes[0] = firstElementCount;
  //deal with the special case of the first element:
  int startingIndex = 0;
  int lshiftAmount = (32 - firstElementCount);
  out[startingIndex] = ((int*)&data[0])[0] << lshiftAmount;
  
  //this loop can be parallelized:
  for (int i=1; i < countData-1; ++i){
    startingIndex = startingIndexes[i] / 32;
    lshiftAmount = (32 - (startingIndexes[i+1]-startingIndexes[i]));
    int rshiftAmount = startingIndexes[i] % 32;
    int writtenBits = 32 - rshiftAmount;
    out[startingIndex] = ((int*) &out[0])[startingIndex] | 
      ( (((unsigned int*) &data[0])[i] << lshiftAmount) >> rshiftAmount);
    if (32 - writtenBits > 0)
      out[startingIndex+1] = ((int*) &out[0])[startingIndex+1] | 
      ( ((unsigned int*) &data[0])[i] << (lshiftAmount + writtenBits));
  }
  
  //deal with the special case of the last element:
  startingIndex = startingIndexes[countData - 1] / 32;
  lshiftAmount = (32 - lastElementCount);
  int rshiftAmount = startingIndexes[countData - 1] % 32;
  int writtenBits = 32 - rshiftAmount;
  out[startingIndex] = ((int*) &out[0])[startingIndex] | 
    ( (((unsigned int*) &data[0])[countData - 1] << lshiftAmount) >> rshiftAmount);
  
  if (32-writtenBits > 0)
    out[startingIndex+1] = ((int*) &out[0])[startingIndex+1] | 
      ( ((unsigned int*) &data[0])[countData - 1] << (lshiftAmount + writtenBits));
      
  startingIndexes[0] = 0;
}
