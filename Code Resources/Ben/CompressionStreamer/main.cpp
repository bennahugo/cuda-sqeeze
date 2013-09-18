#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
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
FILE * fcomp;
FILE * fdecomp;
double totalCompressWriteTime = 0;
double totalDecompressWriteTime = 0;
double totalDiskReadTime = 0;
int main(int argc, char **argv) {
    using namespace std;
    /*gpuCode::initCUDA();
    float* iv = new float[15];
    float* n = new float[15];
    
    iv[0] = 0.4235811f;
    iv[1] = 3.5687154f;
    iv[2] =-1.2547897f;
    iv[3] = 3.5687154f;
    iv[4] =-1.2547897f;
    iv[5] = 3.5687154f;
    iv[6] =-1.2547897f;
    iv[7] = 3.5687154f;
    iv[8] =-1.2547897f;
    iv[9] = 3.5687154f;
    iv[10] =-1.2547897f;
    iv[11] = 3.5687154f;
    iv[12] =-1.2547897f;
    iv[13] = 3.5687154f;
    iv[14] =-1.2547897f;
    
    n[0] = 0.4335811f;
    n[1] = 3.5587154f;
    n[2] =-1.2447897f;
    n[3] = 3.5587154f;
    n[4] =-1.2447897f;
    n[5] = 3.5587154f;
    n[6] =-1.2447897f;
    n[7] = 3.5587154f;
    n[8] =-1.2447897f;
    n[9] = 3.5587154f;
    n[10] =-1.2447897f;
    n[11] = 3.5587154f;
    n[12] =-1.2447897f;
    n[13] = 3.5587154f;
    n[14] =-1.2447897f;
    
    for (int i = 0; i < 15; ++i){
      uint32_t ivElm = *((uint32_t*)&iv[0] + i);
      uint32_t nElm = *((uint32_t*)&n[0] + i);
      uint32_t x = ivElm ^ nElm;
      printf("%d:::",i);
      printBinaryRepresentation((void*)&ivElm,sizeof(uint32_t));
      printf("(%d)",ivElm);
      cout << " ^ ";
      printBinaryRepresentation((void*)&nElm,sizeof(uint32_t));
      printf("(%d)",nElm);
      cout << " = ";
      printBinaryRepresentation((void*)&x,sizeof(uint32_t));
      cout << '\n';
    }
    gpuCode::compressor::initCompressor(iv,15);
    gpuCode::compressor::compressData(n,15,compressCallback);
    gpuCode::releaseCard();
    delete[] iv;
    delete[] n;*/
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
    if (argc >= 6) {
        writeStream = atoi(argv[5]);
        if (writeStream) {
            {
                std::stringstream concat;
                concat << filename << ".comp";
                std::string outName;
                concat >> outName;
                fcomp = fopen(outName.c_str(),"w");
                if (fcomp == NULL) { //failure
                    cout << "FATAL: USER DEMANDED FILE OUTPUT FROM PROGRAM, BUT THE DESTINATION IS FULL OR READ ONLY. GIVING UP." << endl;
                    exit(1);
                }
            }
            {
                std::stringstream concat;
                concat << filename << ".decomp";
                std::string outName;
                concat >> outName;
                fdecomp = fopen(outName.c_str(),"w");
                if (fdecomp == NULL) { //failure
                    cout << "FATAL: USER DEMANDED FILE OUTPUT FROM PROGRAM, BUT THE DESTINATION IS FULL OR READ ONLY. GIVING UP." << endl;
                    exit(1);
                }
            }
        }
    }
    if (argc >= 7){
      useCUDA = atoi(argv[6]);
    }
    cout << omp_get_max_threads() << " CPU Processor Threads available" << endl;
    if (useCUDA)
      gpuCode::initCUDA();
    else
      cout << "WARNING: USER REQUESTED TO SKIP GPGPU RUN" << endl;
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
      timer::tic();
      astroReader::stride data = astroReader::strideFactory::createStride(f,
									  (i+1)*numPagesPerRead  > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : (i+1)*numPagesPerRead,
									  i*numPagesPerRead > f.getDimensionSize(0)-1 ? f.getDimensionSize(0)-1 : i*numPagesPerRead,
									  f.getDimensionSize(1)-1,0,
									  f.getDimensionSize(2)-1,0);	  
      totalDiskReadTime += timer::toc();
      processStride(data);
    } 
    std::cout << "COMPRESSION RATIO: " << (totalCompressSize/(float)origSize) << std::endl;
    std::cout << "COMPRESSED IN " << totalCompressTime << " seconds @ " << 
      origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/totalCompressTime << " GB/s" << std::endl;
    if (!skipDecompression){  
      std::cout << "DECOMPRESSED IN " << totalDecompressTime << " seconds @ " << 
	origSize*sizeof(float)/1024.0f/1024.0f/1024.0f/totalDecompressTime << " GB/s" << std::endl;
    }
    if (writeStream){
      std::cout << "DISK I/O READ TIME (PER STEP): " << totalDiskReadTime << std::endl; // I make the reasonable assumption that an HDF5 system reads the data in blocks --- the same we would have to do it if we read back before decompression
      std::cout << "COMPRESSION DISK I/O WRITE TIME: " << totalCompressTime << std::endl;
      if (!skipDecompression)
	std::cout << "DECOMPRESSION DISK I/O WRITE TIME: " << totalDecompressTime << std::endl;
      fclose(fcomp);
    }
    if (useCUDA)
      gpuCode::releaseCard();
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
  timer::tic();
  if (writeStream){
      fwrite(decompressedData,sizeof(uint32_t),elementCount,fdecomp);
  }
  totalDecompressWriteTime += timer::toc();
}
void compressCallback(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, 
			    uint32_t chunkCount, uint32_t * chunkSizes){
    if (!skipDecompression){ 
      gpuCode::decompressor::decompressData(elementCount,chunkCount,chunkSizes,
 					 compressedResiduals,compressedPrefixes,decompressCallback);
    }
    /*std::cout << "PREFIX ARRAY:" << std::endl;
    for (int i = 0; i < chunkCount; ++i){
      std::cout << ">>" << i << std::endl;
      for (int eI = 0; eI < compressedPrefixIntCounts[i]; ++eI){
	printBinaryRepresentation((void*)&compressedPrefixes[i][eI],sizeof(uint32_t));
	std::cout << std::endl;
      }
    }
    std::cout << "RESIDUAL ARRAY:" << std::endl;
    for (int i = 0; i < chunkCount; ++i){
      std::cout << ">>" << i << std::endl;
      for (int eI = 0; eI < compressedResidualsIntCounts[i]; ++eI){
	printBinaryRepresentation((void*)&compressedResiduals[i][eI],sizeof(uint32_t));
	std::cout << std::endl;
      }
    }*/
    timer::tic();
    if (writeStream){
      for (uint32_t i = 0; i < chunkCount; ++i){
	 fwrite(&chunkSizes[i],sizeof(uint32_t),1,fcomp);
	 fwrite(compressedPrefixes[i],sizeof(uint32_t),compressedPrefixIntCounts[i],fcomp);
	 fwrite(compressedResiduals[i],sizeof(uint32_t),compressedResidualsIntCounts[i],fcomp);
      }
    }
    totalCompressWriteTime += timer::toc();
}

void processStride(const astroReader::stride & data){
    uint32_t tsSize = data.getTimeStampSize();
    float * ts = (float*)_mm_malloc(sizeof(uint32_t)*tsSize,16);
    currentUncompressedData = ts;
    data.getTimeStampData(0,ts);
    
    gpuCode::compressor::initCompressor(ts,tsSize);
    gpuCode::decompressor::initDecompressor(ts,tsSize);
    
    timer::tic();
    if (writeStream){
      fwrite(ts,sizeof(uint32_t),tsSize,fcomp);
    }
    totalCompressWriteTime += timer::toc();
    
    timer::tic();
    if (writeStream){
      fwrite(ts,sizeof(uint32_t),tsSize,fdecomp);
    }
    totalDecompressWriteTime += timer::toc();
    
    for (int t = 1; t <= data.getMaxTimestampIndex() - data.getMinTimestampIndex(); ++t) {
        data.getTimeStampData(t,ts);
        gpuCode::compressor::compressData(ts,tsSize,compressCallback);
    }
    totalCompressTime += gpuCode::compressor::getAccumulatedRunTimeSinceInit();
    totalDecompressTime += gpuCode::decompressor::getAccumulatedRunTimeSinceInit();
    totalCompressSize += gpuCode::compressor::getAccumulatedCompressedDataSize();
    gpuCode::compressor::releaseResources();
    gpuCode::decompressor::releaseResources();
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
  //cout << endl;
}
