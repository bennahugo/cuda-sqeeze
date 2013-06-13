#include <iostream>
#include <string>
#include "AstroReader/file.h"
#include "AstroReader/stride.h"
#include "AstroReader/stridefactory.h"

#define FILENAME "/media/OS/SKA_DATA/kat7_data/1369768145.h5"

int main(int argc, char **argv) {
    using namespace std;
    astroReader::file f(string(FILENAME));
    astroReader::stride data = astroReader::strideFactory::createStride(f,2,0,1,0,10,0); 
    data.print();
    cout << data.getElement(2,1,11).i;
    return 0;
}


