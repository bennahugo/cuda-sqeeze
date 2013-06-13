/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2013  benjamin <email>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "streamcollector.h"
namespace streamCollector{
  void chunkedRead(astroReader::file astroFile, int maxBlockSizeMB, int numPagesPerRead, 
				      void (*callback)(const astroReader::stride &)){
    long maxBlockSizeBytes = maxBlockSizeMB*1024*1024;
    long pageSize = (astroFile.getDimensionSize(1)-1)*(astroFile.getDimensionSize(2)-1)*2*sizeof(float);
    assert(pageSize*numPagesPerRead < maxBlockSizeBytes);
    int numReads = maxBlockSizeBytes / (pageSize*numPagesPerRead);
    for (int i = 0; i < numReads; ++i){
      astroReader::stride data = astroReader::strideFactory::createStride(astroFile,
									  i*(numPagesPerRead+1) > astroFile.getDimensionSize(0)-1 ? astroFile.getDimensionSize(0)-1 : i*(numPagesPerRead+1),
									  i*numPagesPerRead > astroFile.getDimensionSize(0)-1 ? astroFile.getDimensionSize(0)-1 : i*numPagesPerRead,
									  astroFile.getDimensionSize(1)-1,0,
									  astroFile.getDimensionSize(2)-1,0);
      callback(data);
    }
  }
}//namespace streamCollector
