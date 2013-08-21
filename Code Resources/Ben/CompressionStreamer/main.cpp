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
// #include "gpuCode.h"
#include "GPUCompressor/gpuCode.h"
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
bool useCUDA = true;
double totalCompressTime = 0;
double totalDecompressTime = 0;
long totalCompressSize = 0;

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
      if ((skipDecompression = atoi(argv[3])))
	cout << "WARNING: USER REQUESTED TO SKIP DECOMPRESSION" << endl;
    }
    if (argc >= 5){
      if (!skipDecompression){
	if ((skipValidation = atoi(argv[4])))
	  cout << "WARNING: USER REQUESTED TO SKIP VALIDATION" << endl;
      }
    }
    if (argc >= 6){
      writeStream = atoi(argv[5]);
    }
    if (argc >= 7){
      useCUDA = atoi(argv[6]);
    }
    cout << omp_get_max_threads() << " CPU Processor Threads available" << endl;
    if (useCUDA)
      initCUDA();
    
    //Read in chunks:
    long maxBlockSizeBytes = MAX_READ_BUFFER_IN_MB*1024*1024;
    long pageSize = (f.getDimensionSize(1))*(f.getDimensionSize(2))*2*sizeof(float);
    long fileSize = (f.getDimensionSize(0))*pageSize;
    cout << "File Size (bytes): " << fileSize << endl;
    cout << "Page Size (bytes): " << pageSize << endl;
    cout << "Maximum Readible Block (bytes): " << maxBlockSizeBytes << endl;
    assert(pageSize < maxBlockSizeBytes);
    int numPagesPerRead = maxBlockSizeBytes/ (float)pageSize;
    int numReads = ceil(fileSize/(float)fmin(numPagesPerRead*pageSize,f.getDimensionSize(0)*pageSize));    

    cout << "Maximum number of pages per read: " << numPagesPerRead << endl;
    for (int i = 0; i < numReads; ++i){
      std::cout << "Processing file chunk " << i+1 << "/" << numReads << std::endl;
      astroReader::stride data = astroReader::strideFactory::createStride(f,
									  (i+1)*numPagesPerRead  > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : (i+1)*numPagesPerRead,
									  i*numPagesPerRead > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : i*numPagesPerRead,
									  f.getDimensionSize(1)-1,0,
									  f.getDimensionSize(2)-1,0);	  
      processStride(data);
    } 
    std::cout << "COMPRESSION RATIO: " << (totalCompressSize/(float)origSize) << std::endl;
    std::cout << "COMPRESSED IN " << totalCompressTime << " seconds @ " << 
      origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/totalCompressTime << " GB/s" << std::endl;
    if (!skipDecompression){  
      std::cout << "DECOMPRESSED IN " << totalDecompressTime << " seconds @ " << 
	origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/totalDecompressTime << " GB/s" << std::endl;
    }
    return 0;
}
void decompressCallback(uint32_t elementCount, uint32_t * decompressedData){
  using namespace std;
  //Automated test of the compression algorithm. Check decompressed data against original timeslice
  if (!skipValidation){ 
    for (uint32_t i = 0; i < elementCount; ++i){
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
      for (uint32_t i = 0; i < chunkCount; ++i){
	 std::cout << chunkSizes;
	 for (uint32_t p = 0; p < compressedPrefixIntCounts[i]; ++p)
	   std::cout << compressedPrefixes[i][p];
	 for (uint32_t r = 0; r < compressedResidualsIntCounts[i]; ++r)
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
    totalCompressTime += cpuCode::compressor::getAccumulatedRunTimeSinceInit();
    totalDecompressTime += cpuCode::decompressor::getAccumulatedRunTimeSinceInit();
    totalCompressSize += cpuCode::compressor::getAccumulatedCompressedDataSize();
    cpuCode::compressor::releaseResources();
    cpuCode::decompressor::releaseResources();
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
