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


#define FILENAME "/media/OS/SKA_DATA/kat7_data/1369853370.h5"
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
      
      processStride(data);
    }
    return 0;
}
void decompressCallback(uint32_t elementCount, uint32_t * decompressedData){
  using namespace std;
  for (int i = 0; i < elementCount; ++i){
    int checkElement = *(uint32_t *)&currentUncompressedData[i];
    if (decompressedData[i] != checkElement){
      std::cout << "SANITY CHECK FAILED at elem:" << i << std::endl;
       std::cout << "Found:\t" << *(float*)&decompressedData[i] << std::endl;
       std::cout << "Expected:\t" << *(float*)&checkElement << std::endl;
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

    double ratio = (accSize / origSize);
    delete[] ts;
    std::cout << "COMPRESSION RATIO: " << ratio << std::endl;
    accSize = 0;
    
    std::cout << "COMPRESSED IN " << cpuCode::compressor::getAccumulatedRunTimeSinceInit() << " seconds @ " << 
      origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/cpuCode::compressor::getAccumulatedRunTimeSinceInit() << " GB/s" << std::endl;
    std::cout << "DECOMPRESSED IN " << cpuCode::decompressor::getAccumulatedRunTimeSinceInit() << " seconds @ " << 
      origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/cpuCode::decompressor::getAccumulatedRunTimeSinceInit() << " GB/s" << std::endl;
}

void printBinaryRepresentation(void * data, int sizeInBytes){
  using namespace std;
  char * temp = (char *)data;
  for (int i = sizeInBytes - 1; i >= 0; --i){
    for (int b = 7; b >= 0; --b)  
      cout << (0x1 << b & temp[i] ? '1' : '0');
    cout << ' ';
  }
  cout << endl;
}