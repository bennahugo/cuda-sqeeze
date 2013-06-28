#include <iostream>
#include <string>
#include <cstring>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include "AstroReader/file.h"
#include "AstroReader/stride.h"
#include "AstroReader/stridefactory.h"
#include "Compressor/cpuCode.h"
#include "Timer.h"


#define FILENAME "/home/benjamin/1370275467.h5"
#define MAX_READ_BUFFER_IN_MB 1024
void usedBitCountTest(uint32_t * data, int countData, int maxLeadingZeroCount, uint32_t * out);
void printBinaryRepresentation(void * data, int sizeInBytes);
void processStride(const astroReader::stride & data);
void compressCallback(uint32_t elementCount, uint32_t compressedResidualsIntCount, uint32_t * compressedResiduals,
		       uint32_t compressedPrefixIntCount, uint32_t * compressedPrefixes);

unsigned int accSize = 0;
float * currentUncompressedData = NULL;
int main(int argc, char **argv) {
    using namespace std;
    astroReader::file f(string(FILENAME));
    cout << "File dimensions: ";
    for (int i = 0; i < f.getDimensionCount(); ++i)
      cout << f.getDimensionSize(i) << ((i == f.getDimensionCount() -1) ? "\n" : " x ");
    
    //Read in chunks:
    long maxBlockSizeBytes = MAX_READ_BUFFER_IN_MB*1024*1024;
    long pageSize = (f.getDimensionSize(1)-1)*(f.getDimensionSize(2)-1)*2*sizeof(float);
    long fileSize = (f.getDimensionSize(0)-1)*pageSize;
    assert(pageSize < maxBlockSizeBytes);
    int numReads = ceil(fileSize / (float)maxBlockSizeBytes);
    int numPagesPerRead = fileSize / numReads / pageSize;
    for (int i = 0; i < numReads; ++i){
      astroReader::stride data = astroReader::strideFactory::createStride(f,
									  (i+1)*numPagesPerRead  > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : (i+1)*numPagesPerRead,
									  i*numPagesPerRead > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : i*numPagesPerRead,
									  f.getDimensionSize(1)-1,0,
									  f.getDimensionSize(2)-1,0);
      //std::cout << "DATA HERE " << data.getElement(1,0,0).r << std::endl;
      
      processStride(data);
    }
//     float * data1 = new float[13];//{2.43f,0.245f,0.249f,0.409f,3.092f};
//     float * data2 = new float[13];//{2.83f,0.125f,0.289f,0.415f,4.092f};
//     currentUncompressedData = data2;
//     data1[0] = 89.5552f;
//     data1[1] = 0.245f;
//     data1[2] = 0.249f;
//     data1[3] = 0.409f;
//     data1[4] = 3.092f;
//     data1[5] = 3.092f;
//     data1[6] = 3.092f;
//     data1[7] = 3.092f;
//     data1[8] = 3.092f;
//     data1[9] = 3.092f;
//     data1[10] = 3.092f;
//     data1[11] = 3.092f;
//     data1[12] = 2.43f;
//    
//     data2[0] = 89.7276f;
//     data2[1] = 0.125f;
//     data2[2] = 0.289f;
//     data2[3] = 0.415f;
//     data2[4] = 4.092f;
//     data2[5] = 4.092f;
//     data2[6] = 4.092f;
//     data2[7] = 4.092f;
//     data2[8] = 4.092f;
//     data2[9] = 4.092f;
//     data2[10] = 4.092f;
//     data2[11] = 4.092f;
//     data2[12] = 2.83f;
//    
//     for (int i = 0; i < 13; ++i){
//       uint32_t t = ((uint32_t*)&data2[0])[i];
//       printBinaryRepresentation(&t,sizeof(uint32_t));
//     }
//     cpuCode::compressor::initCompressor(data1,13);
//     cpuCode::decompressor::initDecompressor(data1,13);
//     cpuCode::compressor::compressData(data2,13,compressCallback);
//     cpuCode::compressor::releaseResources();
//     cpuCode::decompressor::releaseResources();
//     delete[] data1;
//     delete[] data2;
    return 0;
}
void decompressCallback(uint32_t elementCount, uint32_t * decompressedData){
  using namespace std;
#pragma omp parallel for shared(decompressedData,elementCount,currentUncompressedData)
  for (int i = 0; i < elementCount; ++i){
    int checkElement = *(uint32_t *)&currentUncompressedData[i];
    if (decompressedData[i] != checkElement){
      std::cout << "SANITY CHECK FAILED at elem:" << i << std::endl;
       std::cout << "Found:\t";
       std::cout << *(float*)&decompressedData[i] << std::endl;
       //printBinaryRepresentation(&decompressedData[i],sizeof(uint32_t));
       std::cout << "Expected:\t";
       std::cout << *(float*)&checkElement << std::endl;
       //printBinaryRepresentation(&checkElement,sizeof(uint32_t));
      exit(1);
    }
  }
}
void compressCallback(uint32_t elementCount, uint32_t compressedResidualsIntCount, uint32_t * compressedResiduals,
		       uint32_t compressedPrefixIntCount, uint32_t * compressedPrefixes){
  using namespace std;
   accSize += compressedResidualsIntCount+compressedPrefixIntCount;
   cpuCode::decompressor::decompressData(elementCount,compressedResiduals,compressedPrefixes,decompressCallback);
}

void processStride(const astroReader::stride & data){
    uint32_t tsSize = data.getTimeStampSize();
    float * ts = new float[tsSize];
    currentUncompressedData = ts;
    
    data.getTimeStampData(0,ts);
    cpuCode::compressor::initCompressor(ts,tsSize);
    cpuCode::decompressor::initDecompressor(ts,tsSize);
    accSize += tsSize+1;
    double origSize = tsSize;
    std::cout << "Original Timestamp Size:  " << origSize << std::endl;
    for (int t = 1; t <= data.getMaxTimestampIndex() - data.getMinTimestampIndex(); ++t) {
        data.getTimeStampData(t,ts);
        cpuCode::compressor::compressData(ts,tsSize,compressCallback);
	origSize += tsSize;
    }
    //double delta = Timer::toc();
    double ratio = (accSize / origSize);
    delete[] ts;
    std::cout << "COMPRESSION RATIO: " << ratio << std::endl;
    accSize = 0;
    
    //std::cout << "COMPRESSED IN " << delta << " seconds" << std::endl;
    //std::cout << "THROUGHPUT: " << origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/delta << "GB/s" << std::endl;
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