/*
    Predictive 32-bit IEEE 754 floating point data compressor
    Copyright (C) 2013  benjamin bennahugo@aol.com

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


#ifndef GPUCODE_H
#define GPUCODE_H

#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "exceptions.h"
#include "../timer.h"
namespace gpuCode{ 
  void initCUDA();
  void releaseCard();
  namespace compressor{
    void initCompressor(const float * iv, uint64_t ivLength);
    void compressData(const float * data, uint32_t elementCount,
			    void (*callBack)(uint32_t elementCount, uint32_t * compressedResidualsIntCounts, uint32_t ** compressedResiduals,
			    uint32_t * compressedPrefixIntCounts, uint32_t ** compressedPrefixes, uint32_t chunkCount, uint32_t * chunkSizes));
    double getAccumulatedRunTimeSinceInit();
    uint32_t getAccumulatedCompressedDataSize();
    void releaseResources();
  }
  namespace decompressor{
    void initDecompressor(const float * iv, uint64_t ivLength);
    void decompressData(uint32_t elementCount, uint32_t chunkCount, uint32_t * chunkSizes, 
			uint32_t ** compressedResiduals, uint32_t ** compressedPrefixes, 
			void (*callBack)(uint32_t elementCount, uint32_t * decompressedData));
    double getAccumulatedRunTimeSinceInit();
    uint32_t getAccumulatedDecompressedDataSize();
    void releaseResources();
  }
}

#endif // CPUCODE_H
