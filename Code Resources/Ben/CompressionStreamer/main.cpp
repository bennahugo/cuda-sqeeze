#include <iostream>
#include <string>
#include "AstroReader/file.h"
#include "AstroReader/stride.h"
#include "AstroReader/stridefactory.h"
#include "streamcollector.h"
#include <cstring>

#define FILENAME "/media/OS/SKA_DATA/kat7_data/1369858495.h5"
void printBinaryRepresentation(void * data, int sizeInBytes);
void xorArray(void * out, void * data, void * data2, int sizeInBytes);
void processStride(const astroReader::stride & data);
int main(int argc, char **argv) {
    using namespace std;
    astroReader::file f(string(FILENAME));
    cout << "File dimensions: ";
    for (int i = 0; i < f.getDimensionCount(); ++i)
      cout << f.getDimensionSize(i) << ((i == f.getDimensionCount() -1) ? "\n" : " x ");
    streamCollector::chunkedRead(f,1024,2,processStride);
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

