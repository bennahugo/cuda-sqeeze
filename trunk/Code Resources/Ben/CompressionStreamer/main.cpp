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


#define FILENAME "/media/74A5-DB52/1370275467.h5"
#define MAX_READ_BUFFER_IN_MB 1024
void printBinaryRepresentation(void * data, int sizeInBytes);
void processStride(const astroReader::stride & data);
void compressCallback(uint64_t compressedResidualsIntCount, uint32_t * compressedResiduals,
		       uint64_t compressedPrefixIntCount, uint32_t * compressedPrefixes);
unsigned int accSize = 0;

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
int callbackcount = 0;
void compressCallback(uint64_t compressedResidualsIntCount, uint32_t * compressedResiduals,
		       uint64_t compressedPrefixIntCount, uint32_t * compressedPrefixes){
  using namespace std;
  accSize += compressedResidualsIntCount+compressedPrefixIntCount;
  /*cout << "COMPRESSED PREFIXES:" << endl;
    for (uint64_t i = 0; i < compressedPrefixIntCount; ++i)
      printBinaryRepresentation(&(compressedPrefixes[i]),sizeof(uint32_t));
  cout << "COMPRESSED RESIDUALS:" << endl;
    for (uint64_t i = 0; i < compressedResidualsIntCount; ++i)
      printBinaryRepresentation(&(compressedResiduals[i]),sizeof(uint32_t));*/
}

void processStride(const astroReader::stride & data){
    uint32_t tsSize = data.getTimeStampSize();
    float * ts = new float[tsSize];
    data.getTimeStampData(0,ts);
    cpuCode::compressor::initCompressor(ts,tsSize);
    accSize += tsSize+1;
    Timer::tic();
    for (int t = 1; t <= data.getMaxTimestampIndex() - data.getMinTimestampIndex(); ++t) {
        data.getTimeStampData(t,ts);
        cpuCode::compressor::compressData(ts,tsSize,compressCallback);
    }
    delete[] ts;
    std::cout << "COMPRESSION RATIO: " << (accSize / (float) (tsSize*(data.getMaxTimestampIndex()-data.getMinTimestampIndex()+1))) << std::endl;
    accSize = 0;
    std::cout << "COMPRESSED IN " << Timer::toc() << " seconds" << std::endl;
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
