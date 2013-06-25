#include <iostream>
#include <string>
#include "AstroReader/file.h"
#include "AstroReader/stride.h"
#include "AstroReader/stridefactory.h"
#include "Compressor/cpuCode.h"
#include "streamcollector.h"
#include <cstring>
#include <math.h>
#include <omp.h>

#define FILENAME "/media/OS/SKA_DATA/kat7_data/1369858495.h5"
void printBinaryRepresentation(void * data, int sizeInBytes);
/*void xorArray(void * out, void * data, void * data2, int sizeInBytes);
void processStride(const astroReader::stride & data);
void createBitCountArray(float * data, int countData, int ignoreNumSignificantBits, int maxCount,  int * out);
void createPrefixSumArray(int * counts, int numElements, int * out);
void bitsPack(float * data,int firstElementCount, int lastElementCount, 
	      int * startingIndexes, int * out, int countData, int countOut);
int numberOfBytesNeededToCompress(int * startingIndexes,int countData, int lastCount);
void createParallelPrefixSum(int * counts, int numElements);*/

void compressCallback(uint64_t compressedResidualsIntCount, uint32_t * compressedResiduals,
		       uint64_t compressedPrefixIntCount, uint32_t * compressedPrefixes){
  using namespace std;
  
  cout << "COMPRESSED PREFIXES:" << endl;
    for (uint64_t i = 0; i < compressedPrefixIntCount; ++i)
      printBinaryRepresentation(&(compressedPrefixes[i]),sizeof(uint32_t));
  cout << "COMPRESSED RESIDUALS:" << endl;
    for (uint64_t i = 0; i < compressedResidualsIntCount; ++i)
      printBinaryRepresentation(&(compressedResiduals[i]),sizeof(uint32_t));
}

void parallelPrefixSum(uint32_t * counts, uint32_t numElements) {
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
    float data[10] = {-2.532f,0,0,0,2.532f,2.532f,2.532f,2.532f,2.532f,2.532f};
    float data1[10] = {-2.632f,0,0,0,2.632f,2.632f,2.632f,2.632f,2.632f,2.632f};
    for (int i = 0; i < testCount; ++i){
      uint32_t t = ((uint32_t *)&data[0])[i] ^ ((uint32_t *)&data1[0])[i];
      printBinaryRepresentation(&t, sizeof(uint32_t));
    }
    cpuCode::initCompressor(data,testCount);
    cpuCode::compressData(data1,testCount,compressCallback);
    cpuCode::releaseResources();
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
