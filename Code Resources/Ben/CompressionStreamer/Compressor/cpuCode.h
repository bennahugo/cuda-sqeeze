/*
    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) 2013  <copyright holder> <email>

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


#ifndef CPUCODE_H
#define CPUCODE_H

#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <emmintrin.h>
#include <iostream>

#include "exceptions.h"
#include "../Timer.h"
namespace cpuCode{
  namespace compressor{
    void initCompressor(const float * iv, uint64_t ivLength);
    void compressData(const float * data, uint32_t elementCount,
			    void (*callBack)(uint32_t elementCount, uint32_t compressedResidualsIntCount, uint32_t * compressedResiduals,
			      uint32_t compressedPrefixIntCount, uint32_t * compressedPrefixes));
    void releaseResources();
  }
  namespace decompressor{
    void initDecompressor(const float * iv, uint64_t ivLength);
    void decompressData(const uint32_t elementCount, const uint32_t * compressedResiduals, const uint32_t * compressedPrefixes, 
			void (*callBack)(uint32_t elementCount, uint32_t * decompressedData));
    void releaseResources();
  }
}

#endif // CPUCODE_H
