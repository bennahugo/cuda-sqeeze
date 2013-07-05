#include <iostream>
#include <string>
#include <cstring>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <set>
#include "AstroReader/file.h"
#include "AstroReader/stride.h"
#include "AstroReader/stridefactory.h"
#include "Compressor/cpuCode.h"
#define MAX_READ_BUFFER_IN_MB 1024
void usedBitCountTest(uint32_t * data, int countData, int maxLeadingZeroCount, uint32_t * out);
void printBinaryRepresentation(void * data, int sizeInBytes);
void processStride(const astroReader::stride & data);
void compressCallback(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, uint32_t chunkCount, uint32_t * chunkSizes);

float * currentUncompressedData = NULL;
bool skipDecompression = false;
bool skipValidation = false;
bool writeStream = false;
int main(int argc, char **argv) {
    using namespace std;
    if (argc < 2){
      cout << "FATAL: PLEASE SPECIFY MEERKAT HDF5 FILE LOCATION" << endl;
      exit(1);
    }
    //CPURegisters info = getCPUFeatures();
    string filename(argv[1]);
     astroReader::file f(filename);
    int origSize = 1;
    cout << "File dimensions: ";
    for (int i = 0; i < f.getDimensionCount(); ++i){
      cout << f.getDimensionSize(i) << ((i == f.getDimensionCount() -1) ? "\n" : " x ");
      origSize *= f.getDimensionSize(i);
    }
    if (argc >= 3)
      omp_set_num_threads(atoi(argv[2]));
    if (argc >= 4){
      if (skipDecompression = atoi(argv[3]))
	cout << "WARNING: USER REQUESTED TO SKIP DECOMPRESSION" << endl;
    }
    if (argc >= 5){
      if (!skipDecompression){
	if (skipValidation = atoi(argv[4]))
	  cout << "WARNING: USER REQUESTED TO SKIP VALIDATION" << endl;
      }
    }
    if (argc >= 6){
      writeStream = atoi(argv[5]);
    }
    cout << "Processor Threads Available: " << omp_get_max_threads() << endl;
    //Read in chunks:
    long maxBlockSizeBytes = MAX_READ_BUFFER_IN_MB*1024*1024;
    long pageSize = (f.getDimensionSize(1)-1)*(f.getDimensionSize(2)-1)*2*sizeof(float);
    long fileSize = (f.getDimensionSize(0)-1)*pageSize;
    assert(pageSize < maxBlockSizeBytes);
    int numReads = ceil(fileSize / (float)maxBlockSizeBytes);
    int numPagesPerRead = fileSize / numReads / pageSize;
    for (int i = 0; i < numReads; ++i){
      std::cout << "Processing file chunk " << i+1 << "/" << numReads << std::endl;
      astroReader::stride data = astroReader::strideFactory::createStride(f,
									  (i+1)*numPagesPerRead  > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : (i+1)*numPagesPerRead,
									  i*numPagesPerRead > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : i*numPagesPerRead,
									  f.getDimensionSize(1)-1,0,
									  f.getDimensionSize(2)-1,0);	  
      processStride(data);
    } 
    std::cout << "COMPRESSION RATIO: " << (cpuCode::compressor::getAccumulatedCompressedDataSize()/
      (float)origSize) << std::endl;
    std::cout << "COMPRESSED IN " << cpuCode::compressor::getAccumulatedRunTimeSinceInit() << " seconds @ " << 
      origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/cpuCode::compressor::getAccumulatedRunTimeSinceInit() << " GB/s" << std::endl;
    if (!skipDecompression){  
      std::cout << "DECOMPRESSED IN " << cpuCode::decompressor::getAccumulatedRunTimeSinceInit() << " seconds @ " << 
	origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/cpuCode::decompressor::getAccumulatedRunTimeSinceInit() << " GB/s" << std::endl;
    }
    return 0;
}
void decompressCallback(uint32_t elementCount, uint32_t * decompressedData){
  using namespace std;
  //Automated test of the compression algorithm. Check decompressed data against original timeslice
  if (!skipValidation){ 
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
}
void compressCallback(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, 
			    uint32_t chunkCount, uint32_t * chunkSizes){
    if (!skipDecompression){ 
      cpuCode::decompressor::decompressData(elementCount,chunkCount,chunkSizes,
 					 compressedResiduals,compressedPrefixes,decompressCallback);
    }
    if (writeStream){
      for (int i = 0; i < chunkCount; ++i){
	 std::cout << chunkSizes;
	 for (int p = 0; p < compressedPrefixIntCounts[i]; ++p)
	   std::cout << compressedPrefixes[i][p];
	 for (int r = 0; r < compressedResidualsIntCounts[i]; ++r)
	   std::cout << compressedResiduals[i][r];
      }
    }
}

void processStride(const astroReader::stride & data){
    uint32_t tsSize = data.getTimeStampSize();
    float * ts = (float*)_mm_malloc(sizeof(uint32_t)*tsSize,16);
    currentUncompressedData = ts;
    data.getTimeStampData(0,ts);
    
    cpuCode::compressor::initCompressor(ts,tsSize);
    cpuCode::decompressor::initDecompressor(ts,tsSize);
    for (int t = 1; t <= data.getMaxTimestampIndex() - data.getMinTimestampIndex(); ++t) {
        data.getTimeStampData(t,ts);
        cpuCode::compressor::compressData(ts,tsSize,compressCallback);
    }
    
    _mm_free(ts);
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