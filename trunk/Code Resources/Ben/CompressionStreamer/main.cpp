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
void decompressCallback(uint32_t elementCount, uint32_t * decompressedData);
void decompressFromFile(std::string filename);

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
double totalCompressDiskReadTime = 0;
double totalDecompressDiskReadTime = 0;
uint32_t memoryScaling = 1;

int main(int argc, char **argv) {
    using namespace std; 
    if (argc < 2){
      cout << "FATAL: PLEASE SPECIFY MEERKAT HDF5 FILE LOCATION" << endl;
      exit(1);
    }
    //CPURegisters info = getCPUFeatures();
    string filename(argv[1]);
     astroReader::file f(filename);
    long origSize = 1;
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
      useCUDA = atoi(argv[5]);
    }
    if (argc >= 7) {
        writeStream = atoi(argv[6]);
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
	      if (!skipDecompression){
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
    }

    cout << omp_get_max_threads() << " CPU Processor Threads available" << endl;
    if (useCUDA)
      gpuCode::initCUDA();
    

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
      totalCompressDiskReadTime += timer::toc();
      processStride(data);     
    }
      if (writeStream)
         fclose(fcomp);
    //do the compression from file if the write flag is specified
    if (writeStream && !skipDecompression){
	std::cout << "NOTE: DECOMPRESSING FROM FILE" << std::endl;
	decompressFromFile(filename);
    }
    std::cout << "COMPRESSION RATIO: " << (totalCompressSize/((double)origSize*memoryScaling)) << std::endl;
    std::cout << "COMPRESSED IN " << totalCompressTime << " seconds @ " << 
	      origSize*memoryScaling*sizeof(float)/1024.0f/1024.0f/1024.0f/totalCompressTime << " GB/s" << std::endl;

    if (!skipDecompression){  
      std::cout << "DECOMPRESSED IN " << totalDecompressTime << " seconds @ " << 
	origSize*memoryScaling*sizeof(float)/1024.0f/1024.0f/1024.0f/totalDecompressTime << " GB/s" << std::endl;
    }
    //better to time from terminal:
//     if (writeStream){
//       std::cout << "COMPRESSION DISK I/O READ TIME: " << totalCompressDiskReadTime << std::endl;
//       std::cout << "COMPRESSION DISK I/O WRITE TIME: " << totalCompressTime << std::endl;
//       if (!skipDecompression){
// 	std::cout << "DECOMPRESSION DISK I/O WRITE TIME: " << totalDecompressTime << std::endl;
// 	std::cout << "DECOMPRESSION DISK I/O READ TIME: " << totalDecompressDiskReadTime << std::endl;
// 	fclose(fdecomp);
//       }
//     }
    if (useCUDA)
      gpuCode::releaseCard();
    return 0;
}
/**
 * File-based decompression routine (to compare against other compression utilities)
 * This routine is not invoked if the user does not explicitly ask for output
 */
void decompressFromFile(std::string filename){
    using namespace std;
    std::stringstream concat;
    concat << filename << ".comp";
    std::string outName;
    concat >> outName;
    FILE * compressedFile = fopen(outName.c_str(),"r");
    if (compressedFile == NULL) { //failure
        cout << "FATAL: COULD NOT OPEN COMPRESSED FILE FOR DECOMPRESSION STAGE" << endl;
        exit(1);
    }
    //read the header and the IV:
    uint32_t numElements = 0;
    uint32_t numBlocks = 0;
    timer::tic();
    fread (&numElements,sizeof(uint32_t),1,compressedFile);
    uint32_t * ts = new uint32_t[numElements];
    fread (&numBlocks,sizeof(uint32_t),1,compressedFile);
    fread (ts,sizeof(uint32_t),numElements,compressedFile);
    totalDecompressDiskReadTime += timer::toc();
    uint32_t chunkSize = numElements/numBlocks;
    //write the IV to decompressed file:
    if (useCUDA)
      gpuCode::decompressor::initDecompressor((float*)ts,numElements);
    else
      cpuCode::decompressor::initDecompressor((float*)ts,numElements);
    
    timer::tic();
    fwrite(ts,sizeof(uint32_t),numElements,fdecomp);
    fflush(0);
    totalDecompressWriteTime += timer::toc();
    //read residual counts, prefixes and  residuals block by block:
    while (!feof(compressedFile)){
      uint32_t ** compressedResiduals = new uint32_t*[numBlocks];
      uint32_t ** compressedPrefixes = new uint32_t*[numBlocks];
      uint32_t * chunkSizes = new uint32_t[numBlocks];
      for (uint32_t i = 0; i < numBlocks; ++i){
	uint32_t compressedResidualIntCount = 0;
	timer::tic();
	fread(&compressedResidualIntCount,sizeof(uint32_t),1,compressedFile);
	totalDecompressDiskReadTime += timer::toc();
	compressedResiduals[i] = new uint32_t[compressedResidualIntCount];	
	uint32_t elementsInDataBlock = (((i + 1)*chunkSize <= numElements) ? chunkSize : chunkSize-((i + 1)*chunkSize-numElements));
	uint32_t sizeOfPrefixArray = (elementsInDataBlock * 2) / 32 +
                                 ((elementsInDataBlock * 2) % 32 != 0);
	
	compressedPrefixes[i] = new uint32_t[sizeOfPrefixArray];
	chunkSizes[i] = elementsInDataBlock;
	timer::tic();
	fread(compressedPrefixes[i],sizeof(uint32_t),sizeOfPrefixArray,compressedFile);
	fread(compressedResiduals[i],sizeof(uint32_t),compressedResidualIntCount,compressedFile);
	totalDecompressDiskReadTime += timer::toc();
      }
      if (useCUDA)
	gpuCode::decompressor::decompressData(numElements,numBlocks,chunkSizes,
 					 compressedResiduals,compressedPrefixes,decompressCallback);
      else
	cpuCode::decompressor::decompressData(numElements,numBlocks,chunkSizes,
 					 compressedResiduals,compressedPrefixes,decompressCallback);
      //free residual and prefix stores:
      for (uint32_t i = 0; i < numBlocks; ++i){
	delete [] compressedResiduals[i];
	delete [] compressedPrefixes[i];
      }
      delete [] compressedResiduals;
      delete [] compressedPrefixes;
      delete [] chunkSizes;
    }
    if (useCUDA){
      totalDecompressTime += gpuCode::decompressor::getAccumulatedRunTimeSinceInit();
      gpuCode::decompressor::releaseResources();
    }
    else{
      totalDecompressTime += cpuCode::decompressor::getAccumulatedRunTimeSinceInit();
      cpuCode::decompressor::releaseResources();
    }  
    fclose(compressedFile);
    delete[] ts;
    std::cout << "FINISHED DECOMPRESSION ROUTINE FROM FILE" << std::endl;
}
/**
 * Decompression callback
 * Performs sanity check if not in file output mode
 */
void decompressCallback(uint32_t elementCount, uint32_t * decompressedData){
  using namespace std;
  //Automated test of the compression algorithm. Check decompressed data against original timeslice
  if (!skipValidation && !writeStream){ 
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
      fflush(0);
  }
  totalDecompressWriteTime += timer::toc();
}
/**
 * Compressor callback
 * If user requests compression output all compressed blocks are dumped to disk
 * otherwise if decompression is enabled the blocks are sent strait to the decompressor
 */
void compressCallback(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, 
			    uint32_t chunkCount, uint32_t * chunkSizes){        
    if (writeStream){
      timer::tic();
      for (uint32_t i = 0; i < chunkCount; ++i){
	//chunksizes can be recomputed from number of blocks
	 fwrite(&compressedResidualsIntCounts[i],sizeof(uint32_t),1,fcomp);
	 fwrite(compressedPrefixes[i],sizeof(uint32_t),compressedPrefixIntCounts[i],fcomp);
	 fwrite(compressedResiduals[i],sizeof(uint32_t),compressedResidualsIntCounts[i],fcomp);
	 fflush(0);
      }
      totalCompressWriteTime += timer::toc();
    }
    else{
      if (!skipDecompression){ //decompression from file only decompresses at the end
	if (useCUDA){
	  gpuCode::decompressor::decompressData(elementCount,chunkCount,chunkSizes,
 					 compressedResiduals,compressedPrefixes,decompressCallback);
	}
	else {
	  cpuCode::decompressor::decompressData(elementCount,chunkCount,chunkSizes,
 					 compressedResiduals,compressedPrefixes,decompressCallback);
	}
      }
    }
}
/**
 * Routine initializes the compressor and decompressor and starts encoding (timestamps at a time)
 */
void processStride(const astroReader::stride & data){
    uint32_t tsSize = data.getTimeStampSize()*memoryScaling;
    float * ts = (float*) new float[tsSize];
    currentUncompressedData = ts;
    data.getTimeStampData(0,ts);
    for (uint32_t i = 1; i < memoryScaling; ++i)
      memcpy(ts+i*data.getTimeStampSize(),ts,data.getTimeStampSize()*sizeof(float));
    if (useCUDA){
      gpuCode::compressor::initCompressor(ts,tsSize);
      if (!writeStream)
	gpuCode::decompressor::initDecompressor(ts,tsSize);
    } else {
      cpuCode::compressor::initCompressor(ts,tsSize);
      if (!writeStream)
	cpuCode::decompressor::initDecompressor(ts,tsSize);
    }
    timer::tic();
    if (writeStream){
      fwrite(&tsSize,sizeof(uint32_t),1,fcomp);  //number of elements
      uint32_t blockSize = omp_get_max_threads();
      fwrite(&blockSize,sizeof(uint32_t),1,fcomp); //number of blocks
      fwrite(ts,sizeof(uint32_t),tsSize,fcomp); //iv
      fflush(0);
    }
    totalCompressWriteTime += timer::toc();
    
    for (int t = 1; t <= data.getMaxTimestampIndex() - data.getMinTimestampIndex(); ++t) {
        data.getTimeStampData(t,ts);
	for (uint32_t i = 1; i < memoryScaling; ++i)
	  memcpy(ts+i*data.getTimeStampSize(),ts,data.getTimeStampSize()*sizeof(float));
	if (useCUDA)
	  gpuCode::compressor::compressData(ts,tsSize,compressCallback);
	else
	  cpuCode::compressor::compressData(ts,tsSize,compressCallback);
    }
    if (useCUDA) {
      totalCompressTime += gpuCode::compressor::getAccumulatedRunTimeSinceInit();  
      totalCompressSize += gpuCode::compressor::getAccumulatedCompressedDataSize();
      gpuCode::compressor::releaseResources();
      if (!writeStream){
	totalDecompressTime += gpuCode::decompressor::getAccumulatedRunTimeSinceInit();
	gpuCode::decompressor::releaseResources();
      }
    } else {
      totalCompressTime += cpuCode::compressor::getAccumulatedRunTimeSinceInit();  
      totalCompressSize += cpuCode::compressor::getAccumulatedCompressedDataSize();
      cpuCode::compressor::releaseResources();
      if (!writeStream){
	totalDecompressTime += cpuCode::decompressor::getAccumulatedRunTimeSinceInit();
	cpuCode::decompressor::releaseResources();
      }
    }
    delete[] ts;
}
/**
 * Helper routine to print a binary representation of a memory address
 */
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
